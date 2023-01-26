import os
import torch
import numpy as np
import logging
import argparse
from src.data.read_data import read_and_clean_data
from src.data.get_data import get_dataloaders_seq, get_dataloaders_bow, get_vocab_embeddings
from src.train import train


def main(settings):
    if settings.model_name == "dvae" or settings.model_name == "etm" or settings.model_name == "etm_dirichlet" or settings.model_name == "lda":
        save_path = settings.save_dir + "/" + settings.data_name + "/" + settings.model_name
    elif settings.model_name == "syconntm":
        save_path = settings.save_dir + "/" + settings.data_name + "/" + settings.model_name + "/" + \
                    settings.context_type + "/" + str(settings.context_size) + "/" + str(settings.lambda_)
    elif settings.model_name == "lm":
        save_path = settings.save_dir + "/" + settings.data_name + "/" + settings.model_name + "/" + \
                    settings.context_type + "/" + str(settings.context_size)
    else:
        raise NotImplementedError("Model name not implemented.")

    if settings.preprocess:
        save_path += "/preprocessed"
    else:
        save_path += "/not_preprocessed"

    if not os.path.exists(save_path):
        print(f"Creating directory {save_path}, since it does not exist.")
        os.makedirs(save_path)
    else:
        print(f"Directory {save_path} already exists. Warning. Please change the save directory if it's a mistake.")
    # seed everything
    torch.manual_seed(0)
    np.random.seed(0)
    # initialize logger
    logging.basicConfig(filename=save_path + "/log.txt",
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info(f"Run directory: {save_path}; Data: {settings.data_name}; Model: {settings.model_name}")
    logging.info(3 * "=" + " Loading data " + 3 * "=")
    data = read_and_clean_data(data_name=settings.data_name,
                               max_doc_len=settings.max_doc_len,
                               preprocess=settings.preprocess,
                               return_seq=settings.return_seq,
                               context_size=settings.context_size,
                               context_type=settings.context_type,
                               total_train_docs=settings.total_train_docs,
                               total_val_docs=settings.total_val_docs)
    logging.info(3 * "=" + " Creating context data successful. Now creating PyTorch dataloaders." + 3 * "=")
    train_dl, val_dl, test_dl = get_dataloaders_bow(data=data, settings=settings)
    if settings.return_seq:
        train_dl_seq, val_dl_seq, test_dl_seq = get_dataloaders_seq(data=data, settings=settings)
    else:
        train_dl_seq, val_dl_seq, test_dl_seq = None, None, None
    vocab_embeddings = get_vocab_embeddings(vocab=data["vocab"])
    logging.info(3 * "=" + " Creating PyTorch dataloaders successful. Now training." + 3 * "=")
    vocab = data["vocab"]
    train(model_name=settings.model_name,
          data_name=settings.data_name,
          num_topics=settings.num_topics,
          num_syn_topics=settings.num_syn_topics,
          context_type=settings.context_type,
          context_size=settings.context_size,
          train_dl=train_dl,
          val_dl=val_dl,
          test_dl=test_dl,
          train_dl_seq=train_dl_seq,
          val_dl_seq=val_dl_seq,
          test_dl_seq=test_dl_seq,
          vocab=vocab,
          vocab_embeddings=vocab_embeddings,
          text=data["text"],
          max_epochs=settings.max_epochs,
          max_epochs_lm=settings.max_epochs_lm,
          lr=settings.lr,
          lambda_=settings.lambda_,
          device=settings.device,
          save_path=save_path,
          load_lm_path=settings.load_lm_path,
          top_n=settings.top_n,
          threshold_p=settings.threshold_p,
          dec_type=settings.dec_type,)
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--load_lm_path", type=str, default="/raid/work/username/lm_results_large")
    parser.add_argument("--data_name", type=str, default="rotten_tomatoes")
    parser.add_argument("--generate_data", type=bool, default=False)
    parser.add_argument("--max_doc_len", type=int, default=-1)
    parser.add_argument("--total_train_docs", type=int, default=0)
    parser.add_argument("--total_val_docs", type=int, default=0)
    parser.add_argument("--preprocess", type=str, default="false")
    parser.add_argument("--context_type", type=str, default="asymmetric")
    parser.add_argument("--context_size", type=int, default=1)
    parser.add_argument("--lambda_", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_size_seq", type=int, default=512)
    parser.add_argument("--model_name", type=str, default="syconntm")
    parser.add_argument("--top_n", type=int, default=1)
    parser.add_argument("--threshold_p", type=float, default=0.5)
    parser.add_argument("--dec_type", type=str, default="top_n")
    parser.add_argument("--num_topics", type=int, default=50)
    parser.add_argument("--num_syn_topics", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--max_epochs_lm", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda:0")
    settings = parser.parse_args()

    if settings.total_train_docs == 0:
        print("Going to use all training documents.")
        settings.total_train_docs = None

    if settings.total_val_docs == 0:
        print("Going to use all validation documents.")
        settings.total_val_docs = None

    if settings.preprocess == "false":
        settings.preprocess = False
    else:
        settings.preprocess = True

    if settings.model_name == "syconntm" or settings.model_name == "lm":
        settings.return_seq = True
    else:
        settings.return_seq = False

    if settings.load_lm_path == "none":
        settings.load_lm_path = None

    success = main(settings)

    if success:
        print("Run successful.")
    else:
        print("Run Failed.")
