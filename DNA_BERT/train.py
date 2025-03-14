from data_utils import DoTrain

MODEL_NAME = "zhihan1996/DNA_bert_6"
TOKENIZER_NAME = "zhihan1996/DNA_bert_6"

trainer = DoTrain('CHM13_2995.csv', TOKENIZER_NAME, MODEL_NAME)

#10 epochs
trainer.train(10)

    


