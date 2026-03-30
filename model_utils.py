import cv2
import tempfile
from fairseq import tasks, utils
from fairseq.dataclass.configs import GenerationConfig
from omegaconf import OmegaConf

def predict(video_path, model, cfg, task):
    num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    data_dir = tempfile.mkdtemp()
    tsv_cont = ["/\n", f"test-0\t{video_path}\t{None}\t{num_frames}\t{int(16_000*num_frames/25)}\n"]
    label_cont = ["DUMMY\n"]
    with open(f"{data_dir}/test.tsv", "w") as fo:
        fo.write("".join(tsv_cont))
    with open(f"{data_dir}/test.wrd", "w") as fo:
        fo.write("".join(label_cont))

    gen_subset = "test"
    gen_cfg = GenerationConfig(beam=20)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["task"]["modalities"] = ["video"]
    cfg_dict["task"]["data"] = data_dir
    cfg_dict["task"]["label_dir"] = data_dir
    cfg_dict["task"]["noise_prob"] = 0.0
    cfg_dict["task"]["noise_wav"] = None
    cfg = OmegaConf.create(cfg_dict)
    task = tasks.setup_task(cfg.task)
    task.load_dataset(gen_subset, task_cfg=cfg.task)
    generator = task.build_generator([model], gen_cfg)

    def decode_fn(x):
        dictionary = task.target_dictionary
        symbols_ignore = generator.symbols_to_strip_from_output
        symbols_ignore.add(dictionary.pad())
        return task.datasets[gen_subset].label_processors[0].decode(x, symbols_ignore)

    itr = task.get_batch_iterator(dataset=task.dataset(gen_subset)).next_epoch_itr(shuffle=False)
    sample = next(itr)
    hypos = task.inference_step(generator, [model], sample)
    print(hypos[0][0].keys())
    hypo_tokens = hypos[0][0]['tokens'].int().cpu()
    hypo_scores = hypos[0][0]['score']
    hypo_text = decode_fn(hypo_tokens)
    return hypo_text, hypo_tokens, hypo_scores
