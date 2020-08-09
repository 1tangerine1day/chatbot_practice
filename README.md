# chatbot
## model:

* transformer :

        class Seq2Seq(nn.Module):
            def __init__(self, transbert_encoder, transbert_decoder):
                super().__init__()

                self.transbert_encoder = transbert_encoder
                self.transbert_decoder = transbert_decoder

            def forward(self, src, tgt, teacher_forcing_ratio):
                request_embeddings = self.transbert_encoder(src)
                response_meaning = request_embeddings   
                response = self.transbert_decoder(tgt, response_meaning, teacher_forcing_ratio)
                return response

* tansformer + GRU

        class Seq2Seq(nn.Module):
            def __init__(self, transbert_encoder, transbert_decoder, gru_encoder, gru_decoder, dialog_dnn):
                super().__init__()

                self.transbert_encoder = transbert_encoder
                self.transbert_decoder = transbert_decoder

                self.gru_encoder = gru_encoder
                self.gru_decoder = gru_decoder

                self.dialog_dnn = dialog_dnn

            def forward(self, src, tgt, mask_src, mask_tgt, teacher_forcing_ratio):
                request_embeddings = self.transbert_encoder(src,mask_src)
                request_meaning = self.gru_encoder(request_embeddings)

                if TRAIN_DIALOG:
                    response_meaning = self.dialog_dnn(request_meaning)
                else:
                    response_meaning = request_meaning   

                response_embeddings = self.gru_decoder(request_embeddings, tgt, response_meaning)
                response = self.transbert_decoder(tgt, mask_tgt, response_embeddings, teacher_forcing_ratio)

                return response
## data

Gossiping-QA-Dataset : https://www.kaggle.com/zake7749/pttgossipingcorpus

## Torchviz:
![image](https://github.com/1tangerine1day/chatbot_practice/blob/master/chatbot.png)
