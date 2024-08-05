from transformers import pipeline
nlp_qa = pipeline('question-answering')
sequence = "The traffic began to slow down on Pioneer Boulevard in Los Angeles, making it difficult to get out of the city. However, WBGO was playing some cool jazz, and the weather was cool, making it rather pleasant to be making it out of the city on this Friday afternoon. Nat King Cole was singing as Jo and Maria slowly made their way out of LA and drove toward Barstow. They planned to get to Las Vegas early enough in the evening to have a nice dinner and go see a show."
response  = nlp_qa(context=sequence, question='Where is Pioneer Boulevard ?')
print(response)

from transformers import pipeline
nlp_qa = pipeline('question-answering')
sequence = "DESKVUE, a completely new concept in KVM over IP, allows users to create a personalized workspace in which they can simultaneously monitor and interact with up to 16 systems - physical, virtual, and cloud-based - of their choice. Ideal for control room environments, the Black Box receiver ensures instant, error-free switching between systems with a simple mouse click and enables free positioning of system windows across four 4K screens, including ultrawide curved monitors, while providing unique source information for each connected system. Because the Emerald DESKVUE receiver does away with constant switching between systems, users no longer need to spend 40% to 50% of their time switching between systems and monitoring when tasks are complete or when a system requires interaction. With flexible, concurrent access to more than a dozen systems, all within view and easy reach, users maintain complete situational awareness that allows them to improve efficiency and productivity on a day-to-day basis and to take action instantly whenever needed. Unlike solutions created out of complex and costly equipment integrations, DESKVUE is one small box that does it all, simply and securely over IP. Users tailor their workspace by connecting a single keyboard, mouse, USB 3/2 devices, audio, and up to four 4K monitors. Optionally, one of the four monitors can be 5K. DESKVUE connects to physical systems via Emerald transmitters and virtual machines using RDP, PCoIP, PCoIP ultra."
response  = nlp_qa(context=sequence, question='What is DeskVue?')
print(response)
response  = nlp_qa(context=sequence, question='How many systems can it interact with?')
print(response)
response  = nlp_qa(context=sequence, question='What is it most suitable for?')
print(response)
nlp_ner = pipeline("ner")
print(nlp_ner(sequence))
nlp_qa = pipeline('question-answering', model='google/electra-small-generator', tokenizer='google/electra-small-generator')
response  =  nlp_qa(context=sequence, question='What is DeskVue? ?')
print(response)