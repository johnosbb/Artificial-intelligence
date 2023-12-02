# Large Language Models (LLMs)

- The llama-2 model was released by Meta and they released both the base model and the fine tuned assistant model.
- llama-2-70b - consists of two files. A 140 GB parameters file and a C file consisting of approximately 500 lines of code.
- The magic of this LLM is not in the C code, but embedded in the parameters.
- Model training is a lot more computationally involved than model inference. Inference is using the model to generate output.
- The input to the llama-2 model is approximately 10 TB of crawled internet data. This takes around 6,000 GPUs working 24 hours a day for 12 days. It costs around $2M dollars of compute power. The output of this process is the 140 GB parameter file which we can think of as a sort of 100 times compression of the input data.
- This compression is a lossy compression, we do not have a recoverable copy of the input in the output file.
- These numbers are for llama-2, the equivalent for ChatGPT would be hundreds of millions of dollars.
  
  ![Neural Network](./Resource/NeuralNetwork.png)

- The Neural Network is really just trying to predict the next word in a sequence.
- To understand this take the input of the sentence "The cat sat on the ...". The NN would predict "mat" with a certain probability (97%).
- 
  ![Scaling](./Resource/ScalingLaws.png)

- It can be shown mathematically that there is a very close correlation between compression and prediction. The more accurately we can predict the next word, the more efficiently we can compress the dataset.
- 
 ![How To Train ChatGPT](./Resource/HowToTrainChatGPT.png)
- 
- When we are training the model, we are taking a lot of information and compressing it into the model weights (parameters).
- When we infer with the model we take a word and then get the next word and continually feed these words back into the model.
  
  ![ModelDreams](./Resource/ModelDreams.png)

- The model can dream (or hallucinate) text. It knows for example that in general an ISBN abbreviation is followed by a number. The model is often just following the form of what it knows about.

  ![TransformerNeuralNetworkArchitecture](./Resource/TransformerNeuralNetworkArchitecture.png)

- At the heart of the model is a transformer neural network architecture.
- While we know the mathematics of the model, the 100 billion parameters are dispersed throughout the network. We know how to adjust the model as a whole to make it better at predicting the next word, but we do not know how the billions of parameters collaborate to do this.

 ![Reversal Curse](./Resource/ReversalCurse.png)

- We assume the network builds some form of knowledge database, but we really do not understand how this works. As the reversal curse shows, the database model is not like a traditional model. The information can often only be extracted in a certain direction or certain context.
- We can think of LLMs as inscrutable empirical artifacts. We can evaluate them by giving them inputs and then measuring the outputs. In this respect they are unlike traditional algorithms or engineering processes.
- There is a field called mechanistic interpretabilty which seeks to understand how these models work, but we are still at the forefront of this.

  ![Wikipedia](./Resource/WikipediaExample.png)

- The first stage of training gives us the ability to generate internet documents. This is using very large quantities of unstructured data.

  ![Training](./Resource/TrainingTheAssistant.png)

- The second stage called "fine tuning" is where we can create an assistant model that can answer questions we ask. This stage prioritises quality over quantity.


- This second stage of training uses the same model, but it swaps out the source data with more structured labelled data which is created by users creating specific question and answer combinations. The user crafts a question and then crafts what would be the ideal answer for that question.

  ![Labelling Instructions](./Resource/LabellingInstructions.png)

- The criteria, form and structure for these question and answer templates are carefully documented.

- The volume of these second stage inputs is far less, often in the 100K range, but they are of very high quality.
- After this stage we have a model that knows how to answer a question in the style of the higher quality "question - answer" labelled data. It may not have seen the specific answer in the second stage training, but it can at least use the form of answering a question learned in that second stage.
- When the model assist us it is using the data from the first stage but forming its responses in the style and format it learned during the second stage.
- The power of having the llama 2 model available is that the really expensive part is done by Meta. We can take the model and do our own fine tuning stage.
- There is an optional stage three that we can also employ. This stage uses comparisons. It is often easier to compare two answers rather than write an answer.

 ![Comparisons](./Resource/Comparisons.png)

- In the comparison stage we can use the assistant model from stage two to generate candidate answers and we then select the one that we feel is best. This is often easier than generating an answer. This is often called Reinforcement Learning fromHuman Feedback (RHF).

 ![Collaboration](./Resource/Collaboration.png)


- Increasingly human machine collaboration is being used in labelling.
- The current LLMs are rated in the Chatbot arenas.
- Scaling Laws: With just two variables __N__ the number of parameters in the network and __D__ the amount of text we train on we can predict the predictive accuracy of the model. So to improve performance we can simply increase these variables with no fear of a plateau.
- While we are not solely focused on word prediction accuracy we know there is a direct correlation between word prediction accuracy and a range of other bench mark tests.
- When we ask ChatGPT a question similar to the one shown in the image above the LLM will carryout a search, retrieve the data (much as a human would using a browser) and then uses it assistant to format in the way we requested.
- If we ask it to compute or impute from existing data, chatgpt will use a calculator.
- If we ask ChatGPT to plot the data then it will create a python shell and use mathplotlib to plot the data.
- We can extend our request and chatGPT will respond by writing the additional code to create the modified plot.
- This type of tool use is a major aspect of how these models are evolving.
- We can also ask it to Generate an Image based on the previous context (Dali is used in this example).
- In this instance ChatGPT can generate a html based website based on a sketch. This means we can, in the future, we can input both images and text in our interaction.
- ChatGPT can also take audio as an input.
- At the moment LLMs only have system 1 thinking. The challenge is to get LLMs to use system 2 thinking. We would like to be able to allow ChatGPT the time to prioritize accuracy over time. This would require the LLM to create a tree of thoughts and reflect, evaluate different options and reframe the solutions.
- Ideally we would like the development of LLMs to emulate that of AlphaGo. AlphasGo first learned by imitating humans and then in the second stage it learned by self-improvement using a rewards system. What is the equivalent for LLMs of AlphaGo's stage 2.
- How do we customise LLMs so that users can tailor them to specific domains.
- We should think of LLMs as a kernel or OS which uses existing tools and services.
- There are a number of security challenges facing LLMs. We can Jailbreak the model by carefully selecting our prompts. There are a large selection of Jailbreaks available.
- In this example with claude we simply use base64 encoded data of the same question. Claude has learned to refuse harmful queries in English.
- In this example a single universal transferable suffix appended to any prompt can jailbreak the model.
- In this example the Panda Model with carefully embedded noise pattern with structured data can break the model.
- Prompt injections can be embedded in an image. It could be invisible to the human eye.
- In this example one of the web pages that bing had referenced included a prompt injection attack. The text on this new page instructs the LLM to forget the previous prompt and replace it with the new nefarious one.
- In this instance the google doc contains a prompt injection attack, bard is hijacked and encodes personal information into an image URL. If the attacker controls the server then they can retrieve the personal data.
- In a data poising attack the attacker tries to hide a carefully crafted text with a custom trigger phrase. This text attempts to poison the model.

