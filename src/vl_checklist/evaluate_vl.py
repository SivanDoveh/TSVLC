from vl_checklist.utils import chunks
from vl_checklist.data_loader import DataLoader
from tqdm import tqdm
import yaml
import os
import random
import time
import json
import open_clip as clip
from PIL import Image
import torch
import logging
from torch import distributed as dist

def is_master(args):
    return (not args.distributed) or args.rank == 0


def EvaluateAllVL(model, preprocess_val, epoch, args, writer):
    model.eval()

    vl_eval = Evaluate(config_file="vl_checklist/configs/clip_all_obj.yaml", model=model,
                           preprocess_val=preprocess_val, epoch=epoch, args=args, tb_writer=writer)
    vl_eval.start()

    vl_eval = Evaluate(config_file="vl_checklist/configs/clip_all_attribute.yaml", model=model,
                       preprocess_val=preprocess_val, epoch=epoch, args=args, tb_writer=writer)
    vl_eval.start()

    vl_eval = Evaluate(config_file="vl_checklist/configs/clip_all_rel.yaml", model=model,
                       preprocess_val=preprocess_val, epoch=epoch, args=args, tb_writer=writer)
    vl_eval.start()

    vl_eval = Evaluate(config_file="vl_checklist/configs/clip_all_rel_spatial.yaml", model=model,
                       preprocess_val=preprocess_val, epoch=epoch, args=args, tb_writer=writer)
    vl_eval.start()

    m = json.load(open('training/' + 'corpus.json'))
    path = os.path.join(args.vl_checklist_accuracy_jsons_folder, args.name)
    score_list = []
    for ind, item in enumerate(m.keys()):
        data_num = len(m[item].keys())
        data_score = []
        for data in m[item].keys():
            score = 0
            file_num = len(m[item][data])
            for file in m[item][data]:
                json_name = os.path.join(path,f"{file}_{epoch}.json")
                if not os.path.exists(json_name):
                    print(f"{file}_{epoch}.json has not been evaluated. exp name: {args.name}")
                else:
                    m1 = json.load(open(json_name))
                    score += m1["total_acc"]

            data_score.append(score/file_num)
        score_list.append(sum(data_score)/data_num)

    test_names = ['O-Large', 'O-Medium', 'O-Small', 'O-Center', 'O-Mid', 'O-Margin', 'A-Color', 'A-Material', 'A-Size',"A-State", "A-Action", "R-action", "R-spatial",]
    print(test_names)
    print(f'{args.name} {score_list}')


class Evaluate(object):
    def __init__(self, config_file, model,preprocess_val,epoch,args,tb_writer=None) -> None:
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        self.log_dir =  os.path.join(args.logs, args.name)
        m = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
        self.batch_size = m["BATCH_SIZE"]
        self.model = model.module if args.distributed else model

        self.max_num = m["MAX_NUM"]
        self.data_names = m["DATA"]["TEST_DATA"]
        self.task = m["TASK"]
        self.types = m["DATA"]["TYPES"]
        self.dir = m["OUTPUT"]["DIR"]
        self.sample_num = m["OUTPUT"]["NUM"]
        self.model_name = self.model.model_name
        self.preprocess_val_clip = preprocess_val
        self.epoch = epoch
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tb_writer = tb_writer
        self.config_file = config_file
        self.saliency_layer = "layer4"


    def start(self):
        avg = 0
        for data_type in self.types:
            results = self.eval(data_type=data_type)
        #     avg += results
        # avg = avg / len(self.types)
        # if self.args.save_logs:
        #     if self.tb_writer is not None:
        #         self.tb_writer.add_scalar(f"val/{self.types[0].split('/')[0]}_avg_eval", avg, self.epoch)
        #         logging.info(
        #             f" AVG {self.epoch}: {self.types[0].split('/')[0]}_avg_eval {avg}")

    def clip_model_wrapper(self, images, texts):
        probs = []
        for i, chunk_i in enumerate(chunks(images, self.batch_size)):
            for j in range(len(chunk_i)):
                image = self.preprocess_val_clip(Image.open(chunk_i[j])).unsqueeze(0).to(self.device)
                text = clip.tokenize(texts[j]).to(self.device)
                with torch.no_grad():
                    image_features, text_features, logit_scale = self.model(image, text)
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    probs.extend(logits_per_image.cpu().numpy())
        return {"probs": probs}

    def eval(self, data_type):
        max_number = self.max_num
        d = DataLoader(self.data_names,self.args, data_type, self.task)
        results = {}
        index = 0

        if self.task == 'itc':
            for name in d.data:
                path = os.path.join(self.args.vl_checklist_accuracy_jsons_folder, self.args.name)
                file_name = data_type.replace("/", "_")
                if os.path.exists(os.path.join(path, f'{file_name}_{name}_{self.epoch}.json')):
                    continue
                if not is_master(self.args):
                    continue
                sample_true = []
                sample_false = []
                num_t, num_f = 0, 0
                if max_number:
                    d.data[name] = d.data[name][:int(max_number / 2)]
                starttime = time.time()
                for batch in tqdm(chunks(d.data[name], self.batch_size), desc="Progress", ncols=100,
                                  total=int(len(d.data[name]) / self.batch_size)):
                    images = [z["path"] for z in batch]
                    texts_pos = [z['texts_pos'][0] for z in batch]
                    texts_neg = [z['texts_neg'][0] for z in batch]

                    result_pos = self.clip_model_wrapper(images, texts_pos)
                    result_neg = self.clip_model_wrapper(images, texts_neg)

                    result_t1 = zip(result_pos["probs"], result_neg["probs"])
                    result_tmp = list(result_t1)


                    for i in range(len(result_tmp)):
                        index = index + 1
                        if result_tmp[i][0][0] > result_tmp[i][1][0]:
                            sample_true.append({"img_path": images[i], "pos_score": float(round(result_tmp[i][0][0], 4)),
                                                "pos_txt": texts_pos[i], "neg_score": float(round(result_tmp[i][1][0], 4)),
                                                "neg_txt": texts_neg[i], "result": "correct"})
                            num_t += 1


                        else:
                            sample_false.append({"img_path": images[i], "pos_score": float(round(result_tmp[i][0][0], 4)),
                                                 "pos_txt": texts_pos[i], "neg_score": float(round(result_tmp[i][1][0], 4)),
                                                 "neg_txt": texts_neg[i], "result": "incorrect"})
                            num_f += 1

                endtime = time.time()
                accuracy = float(num_t) / (num_t + num_f)
                results[name] = round(accuracy, 4)
                file_name = data_type.replace("/", "_")
                # try:
                #     path = os.path.join(self.args.vl_checklist_accuracy_jsons_folder, self.args.resume.split('/')[-3])
                # except:
                path = os.path.join(self.args.vl_checklist_accuracy_jsons_folder, self.args.name)
                os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, f'{file_name}_{name}_{self.epoch}.json'), 'w',
                          encoding='utf-8') as f:
                    json.dump({"total_acc": round(accuracy, 4), "number_of_data": len(d.data[name]),
                               "model_name": self.model_name, "task": self.task, "eval_time": endtime - starttime}, f)

                logging.info(
                    f"Eval {name} VL Epoch: {self.epoch} {data_type}_eval: {round(accuracy, 4)}")

                if self.args.save_logs:
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar(f"val/{name}/{data_type}_eval", round(accuracy, 4), self.epoch)


        # if self.args.save_logs and is_master(self.args):
        #     if self.tb_writer is not None:
        #         both_res=0
        #         for k in results.keys():
        #             both_res += results[k]
        #         both_res = both_res/results.keys().__len__()
        #         self.tb_writer.add_scalar(f"val/both/{data_type}_eval", both_res, self.epoch)
        #
        #         return both_res
        return 0


