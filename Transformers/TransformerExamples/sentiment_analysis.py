from transformers import pipeline

nlp = pipeline("sentiment-analysis")

print(nlp("If you sometimes like to go to the movies to have fun , Wasabi is a good place to start ."),"If you sometimes like to go to the movies to have fun , Wasabi is a good place to start .")
print(nlp("Effective but too-tepid biopic."),"Effective but too-tepid biopic.")


def classify(sequence,M):
   #DistilBertForSequenceClassification(default model)
    nlp_cls = pipeline('sentiment-analysis')
    if M==1:
      print(nlp_cls.model.config)
    return nlp_cls(sequence)


seq=3
if seq==1:
  sequence="The battery on my Model9X phone doesn't last more than 6 hours and I'm unhappy about that."
if seq==2:
  sequence="The battery on my Model9X phone doesn't last more than 6 hours and I'm unhappy about that. I was really mad! I bought a Moel10x and things seem to be better. I'm super satisfied now."
if seq==3:
  sequence="The customer was very unhappy"
if seq==4:
  sequence="The customer was very satisfied"
print(sequence)
M=0 #display model cofiguration=1, default=0
CS=classify(sequence,M)
print(CS)