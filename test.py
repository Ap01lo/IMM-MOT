import yaml, argparse, time, os, json, multiprocessing
from dataloader.nusc_loader import NuScenesloader
from nuscenes.nuscenes import NuScenes
from tracking.nusc_tracker import Tracker
from tqdm import tqdm
import pdb
import logging
import logging.handlers
import datetime
import time
import shutil,gc


parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--process', type=int, default=16)
# paths
localtime = '_'.join(time.asctime(time.localtime(time.time())).split(' '))
parser.add_argument('--nusc_path', type=str, default='/data/nuscenes/NuScenes_trainval')
parser.add_argument('--config_path', type=str, default='config/nusc_config.yaml')
parser.add_argument('--detection_path', type=str, default='data/detector/val/')
parser.add_argument('--first_token_path', type=str, default='data/utils/first_token_table/trainval/nusc_first_token.json')
parser.add_argument('--result_path', type=str, default='result/' + localtime)
parser.add_argument('--eval_path', type=str, default='eval_result3/')
args = parser.parse_args()

args.eval_path = args.result_path

def main(result_path, token, process, nusc_loader):
    # PolyMOT modal is completely dependent on the detector modal
    result = {
        "results": {},
        "meta": {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
    }
    # tracking and output file
    nusc_tracker = Tracker(config=nusc_loader.config)
    for frame_data in tqdm(nusc_loader, desc='Running', total=len(nusc_loader) // process, position=token):
        if process > 1 and frame_data['seq_id'] % process != token:
            continue
        sample_token = frame_data['sample_token']
        # track each sequence
        nusc_tracker.tracking(frame_data)
        """
        only for debug
        {
            'np_track_res': np.array, [num, 17] add 'tracking_id', 'seq_id', 'frame_id'
            'box_track_res': np.array[NuscBox], [num,]
            'no_val_track_result': bool
        }
        """
        # output process
        sample_results = []
        if 'no_val_track_result' not in frame_data:
            for predict_box in frame_data['box_track_res']:
                box_result = {
                    "sample_token": sample_token,
                    "translation": [float(predict_box.center[0]), float(predict_box.center[1]),
                                    float(predict_box.center[2])],
                    "size": [float(predict_box.wlh[0]), float(predict_box.wlh[1]), float(predict_box.wlh[2])],
                    "rotation": [float(predict_box.orientation[0]), float(predict_box.orientation[1]),
                                 float(predict_box.orientation[2]), float(predict_box.orientation[3])],
                    "velocity": [float(predict_box.velocity[0]), float(predict_box.velocity[1])],
                    "tracking_id": str(predict_box.tracking_id),
                    "tracking_name": predict_box.name,
                    "tracking_score": predict_box.score,
                }
                sample_results.append(box_result.copy())

        # add to the output file
        if sample_token in result["results"]:
            result["results"][sample_token] = result["results"][sample_token] + sample_results
        else:
            result["results"][sample_token] = sample_results

    # sort track result by the tracking score
    for sample_token in result["results"].keys():
        confs = sorted(
            [
                (-d["tracking_score"], ind)
                for ind, d in enumerate(result["results"][sample_token])
            ]
        )
        result["results"][sample_token] = [
            result["results"][sample_token][ind]
            for _, ind in confs[: min(500, len(confs))]
        ]
    del nusc_loader
    gc.collect()
    # write file
    if process > 1:
        json.dump(result, open(result_path +'/'+ str(token) + ".json", "w"))
    else:
        json.dump(result, open(result_path + "/results.json", "w"))


def eval(result_path, eval_path, nusc_path):
    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.eval.common.config import config_factory as track_configs
    cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEval(
        config=cfg,
        result_path=result_path,
        eval_set="val",
        output_dir=eval_path,
        verbose=True,
        nusc_version="v1.0-trainval",
        nusc_dataroot=nusc_path,
    )
    print("result in " + result_path)
    metrics_summary = nusc_eval.main()


if __name__ == "__main__":
    # time.sleep(7200)

    detections = os.listdir(args.detection_path)
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.Loader)
    if config['preprocessing']['DBSE']:
        nusc = NuScenes(version='v1.0-trainval', dataroot=args.nusc_path, verbose=True)
    else:
        nusc = None
    # logger
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    rf_handler = logging.handlers.TimedRotatingFileHandler('./log/all.log', when='D',interval=1, atTime=datetime.time(0, 0, 0, 0))
    rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(rf_handler)
    logger.info("************************NEW SESSION START************************")

    for detection in detections:
        detection = args.detection_path + detection

        localtime = '_'.join(time.asctime(time.localtime(time.time())).split(' '))
        args.result_path = 'result/'+localtime
        args.eval_path = args.result_path

        logger.info('-'*30)
        logger.info(f'MOT START ---- detection {detection} -- result at {args.result_path}')

        os.makedirs(args.result_path, exist_ok=True)
        os.makedirs(args.eval_path, exist_ok=True)

        # load and keep config
        
        valid_cfg = config
        json.dump(valid_cfg, open(args.eval_path + "/config.json", "w"))
        print('writing config in folder: ' + os.path.abspath(args.eval_path))

        # load dataloader
        nusc_loader = NuScenesloader(detection,
                                    args.first_token_path,
                                    nusc,
                                    config)
        print('writing result in folder: ' + os.path.abspath(args.result_path))

        if args.process > 1:
            result_temp_path = args.result_path + '/temp_result'
            os.makedirs(result_temp_path, exist_ok=True)
            pool = multiprocessing.Pool(args.process)
            for token in range(args.process):
                pool.apply_async(main, args=(result_temp_path, token, args.process, nusc_loader))
            pool.close()
            pool.join()
            results = {'results': {}, 'meta': {}}
            # combine the results of each process
            for token in range(args.process):
                result = json.load(open(os.path.join(result_temp_path, str(token) + '.json'), 'r'))
                results["results"].update(result["results"])
                results["meta"].update(result["meta"])
            json.dump(results, open(args.result_path + '/results.json', "w"))
            print('writing result in folder: ' + os.path.abspath(args.result_path))
        else:
            main(args.result_path, 0, 1, nusc_loader)
            print('writing result in folder: ' + os.path.abspath(args.result_path))

        logger.info(f'MOT END ---- detection {detection} -- result at {args.result_path}')
        # eval result
        logger.info(f'EVAL START ---- detection {detection} -- result from {args.result_path} -- eval at {args.eval_path}')
        eval(os.path.join(args.result_path, 'results.json'), args.eval_path, args.nusc_path)
        logger.info(f'EVAL END ---- detection {detection} -- result from {args.result_path} -- eval at {args.eval_path}')
        logger.info('-'*30)
        shutil.rmtree(args.result_path + '/temp_result')
        