import os
import torch
import logging
import pickle
from src.training.dvae import DVAETrainer
from src.training.etm import ETMTrainer
from src.training.etm_dirichlet import ETMDirichletTrainer
from src.training.lm import LMTrainer
from src.training.syconntm import SyConNTMTrainer
from src.training.lda import LDATrainer

logger = logging.getLogger(__name__)


def train(model_name: str,
          data_name: str,
          num_topics: int,
          num_syn_topics: int,
          context_type: str,
          context_size: int,
          train_dl: torch.utils.data.DataLoader,
          val_dl: torch.utils.data.DataLoader,
          test_dl: torch.utils.data.DataLoader,
          train_dl_seq: torch.utils.data.DataLoader,
          val_dl_seq: torch.utils.data.DataLoader,
          test_dl_seq: torch.utils.data.DataLoader,
          vocab: dict,
          vocab_embeddings: torch.Tensor,
          text: list,
          max_epochs: int,
          max_epochs_lm: int,
          lr: float,
          lambda_: float,
          device: str,
          save_path: str,
          load_lm_path: str,
          top_n: int,
          threshold_p: float,
          dec_type: str) -> None:
    if model_name == 'dvae':
        trainer = DVAETrainer(num_topics=num_topics,
                              vocab=vocab,
                              train_dl=train_dl,
                              val_dl=val_dl,
                              test_dl=test_dl,
                              max_epochs=max_epochs,
                              device=device,
                              text=text,
                              lr=lr,
                              save_path=save_path)
        trainer.run()

    elif model_name == 'etm_dirichlet':
        trainer = ETMDirichletTrainer(num_topics=num_topics,
                                      vocab=vocab,
                                      vocab_embeddings=vocab_embeddings,
                                      train_dl=train_dl,
                                      val_dl=val_dl,
                                      test_dl=test_dl,
                                      max_epochs=max_epochs,
                                      device=device,
                                      text=text,
                                      lr=lr,
                                      save_path=save_path)
        trainer.run()

    elif model_name == 'etm':
        trainer = ETMTrainer(num_topics=num_topics,
                             vocab=vocab,
                             vocab_embeddings=vocab_embeddings,
                             train_dl=train_dl,
                             val_dl=val_dl,
                             test_dl=test_dl,
                             max_epochs=max_epochs,
                             device=device,
                             text=text,
                             lr=lr,
                             save_path=save_path)
        trainer.run()

    elif model_name == 'lm':
        trainer = LMTrainer(num_topics=num_topics,
                            vocab=vocab,
                            vocab_embeddings=vocab_embeddings,
                            context_type=context_type,
                            context_size=context_size,
                            train_dl=train_dl_seq,
                            val_dl=val_dl_seq,
                            test_dl=test_dl_seq,
                            max_epochs=max_epochs_lm,
                            device=device,
                            text=text,
                            lr=lr,
                            save_path=save_path)
        trainer.run()

    elif model_name == 'syconntm':
        # first we train the LM
        load_lm_path = os.path.join(load_lm_path, data_name, 'lm', context_type, str(context_size), 'not_preprocessed')
        lm_path = os.path.join(load_lm_path, 'lm.pt')
        if not (os.path.exists(lm_path)):
            logger.info('Training the LM')
            trainer = LMTrainer(num_topics=num_topics,
                                vocab=vocab,
                                vocab_embeddings=vocab_embeddings,
                                context_type=context_type,
                                context_size=context_size,
                                train_dl=train_dl_seq,
                                val_dl=val_dl_seq,
                                test_dl=test_dl_seq,
                                max_epochs=max_epochs_lm,
                                device=device,
                                text=text,
                                lr=lr,
                                save_path=save_path)
            syntax_vector, content_vector = trainer.run()
        else:
            logger.info('Loading the LM results since they are already trained')
            if dec_type == 'top_n':
                content_vector_path = os.path.join(load_lm_path, f'content_vector_top_{top_n}.pkl')
                syntax_vector_path = os.path.join(load_lm_path, f'syntax_vector_top_{top_n}.pkl')
                content_vector = pickle.load(open(content_vector_path, 'rb'))
                syntax_vector = pickle.load(open(syntax_vector_path, 'rb'))
            elif dec_type == 'threshold_p':
                content_vector_path = os.path.join(load_lm_path, f'content_vector_threshold_{threshold_p}.pkl')
                syntax_vector_path = os.path.join(load_lm_path, f'syntax_vector_threshold_{threshold_p}.pkl')
                content_vector = pickle.load(open(content_vector_path, 'rb'))
                syntax_vector = pickle.load(open(syntax_vector_path, 'rb'))
        # now the topic model is trained on the whole dataset
        trainer = SyConNTMTrainer(num_topics=num_topics,
                                  num_syn_topics=num_syn_topics,
                                  vocab=vocab,
                                  train_dl=train_dl,
                                  val_dl=val_dl,
                                  test_dl=test_dl,
                                  syntax_vector=syntax_vector,
                                  content_vector=content_vector,
                                  max_epochs=max_epochs,
                                  device=device,
                                  text=text,
                                  lr=lr,
                                  lambda_=lambda_,
                                  save_path=save_path)
        trainer.run()

    elif model_name == 'lda':
        trainer = LDATrainer(num_topics=num_topics,
                             text=text,
                             train_dl=train_dl,
                             vocab=vocab,
                             save_path=save_path)
        trainer.run()

    else:
        logger.critical(f"Model name {model_name} not supported.")
        raise ValueError(f"Model name {model_name} not supported.")
