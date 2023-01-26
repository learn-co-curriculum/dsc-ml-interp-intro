# Machine Learning Interpretability - Introduction

## Introduction

If you feel like the term "machine learning interpretability" is a little tricky to define, you are not alone. According to a recent blog post on [Machine Learning Carnegie-Melon University \(ML-CMU\)](https://blog.ml.cmu.edu/2020/08/31/6-interpretability/#:~:text=Interpretability%20is%20difficult,can%20be%20evaluated.), there is no clear consensus among the machine learning community as to what we mean by interpretability and what we expect to interpret from our models. 

Instead of getting caught up in the semantics of what interpretability means, we should instead focus on the goals of our specific machine learning project and select the models and metrics that will reveal the information we are interested in most effectively.

In this lesson, we will discuss different aspects of interpretability and how we can consider it during model selection. We will also discuss different approaches to generating useful results from machine learning models.   

## Objectives

You will be able to:

* Describe how interpretability supports the goals of a machine learning project
* Understand the high level differences between "black box" and "white box" models
* Explain interpretability in the context of machine learning

## Interpretability and Machine Learning Goals

Machine learning is a tool that can be applied to broad array of applications. Typically, the fundamental goal of machine learning is to predict an outcome to inform a decision. Different projects achieve the goal of predictive modeling in various ways with methodologies that are tailored to their individual use case.

### Machine Learning Goals

#### Support Human Decisions
In most cases, the goal of a machine learning project is to simulate various outcomes of what-if scenarios using historical data to __support humans with insights to make major decisions__. One type of framework that exhibits this goal are Clinical Decision Support (CDS) applications like __Watson Health's Micromedex__, which we will discuss in more detail in the upcoming section.

#### Automate Human Decisions
In other cases, the goal is to __train the machine itself to make a decision__ based on exposure to many examples. One framework that exhibits this goal are __Natural Language Generation__ applications, that can be found everywhere from predictive text on your phone to advanced text generation algorithms like ChatGPT. 

You might notice that the first goal, __support humans with insights to make major decisions__ limits the impact of the model's decision because it can be accepted or rejected by the clinician. For the second goal, __train the machine itself to make a decision__, the model's decision has more of an impact on the final result of the text that is output. Since the stakes of these use cases differ greatly, it makes sense that the goal is __support__ and not __replace__ the clinician to maintain a high level of safety. 

### The Cost and Benefits of Decision Support
In the sections below, we examine two applications of machine learning in medicine, Clinical Decision Support and Computer Aided Detection. In both of these applications, machine learning provides support to clinicians with a goal of increasing the speed of patient care. However, there are costs and benefits related to introducing machine learning models in the context of medical decisions that are important to assess when evaluating the efficacy of a machine learning project.

#### Clinical Decision Support (CDS)
__Clinical Decision Support (CDS)__ is an application of artificial intelligence implemented by software like [__Watson Health Micromedex__](https://www.ibm.com/watson-health/solutions/clinical-decision-support) in hospitals. Micromedex and other CDS systems provide advice for pharmacists and doctors regarding complications, contraindications, and treatment options by leveraging patient medical data with vast medical knowledge bases. In the past, providers would examine patient health records individually, conduct research, and call other providers for an opinion before presenting available interventions to patients. This was a time consuming process that interfered with patient care. Micromedex simplifies this process by integrating medical knowledge bases with patient health records to provide customized recommendations to providers on demand. 

> In this case, the goal of the artificial intelligence is to __provide the clinician with reliable information, like possible complications or contraindications based on the patient's medical history__, to aid the clinician in selecting appropriate options. For the application to be effective, the results need be relevant to the target patient. False positive relevant search results increase the amount of time the clinician has to sort through results. 


#### Computer Aided Detection
__Computer Aided Detection (CAD)__ is an application of artificial intelligence that is implemented in various diagnostic imaging settings. In the context of breast cancer screenings, clinicians use CAD as a "second set of eyes" to quickly detect signs of cancer in an image that might not be perceivable by the human eye. This leads to earlier detection, which is correlated with better outcomes for patients. [__There is some debate regarding how useful CAD really is__](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6670274/) in improving patient outcomes in its current state, due to a non-trivial occurrence of false positives. While it is possible that CAD can flag instances of cancer more rapidly than a clinician can alone, false positives can a negative impact on the patient that has to endure additional unnecessary tests entire and health care system by wasting resources needed by other patients. Efforts are underway to improve CAD, especially in the area of breast cancer screenings. 

> In this case, the goal of the artificial intelligence is to __override a potential false negative and change the doctor's recommendation__. This is considered an acceptable trade-off, because of the benefits of early detection are believed to outweigh the drawbacks of a false positive.

### Interpretability and Improving Decison Support
While neither CDS or CAD are perfect, they do provide some enhancements to the diagnostic process, that can extend the abilities of clinicians when used effectively. These applications of machine learning in medicine demonstrates how different problems can have a common desired outcome (diagnosing and treating patients more quickly) but require different approaches. In the case of CDS, __precision__ or the specificity of the algorithm would be more important to our overall goal than __accuracy__ alone. On the other hand, CAD prioritizes __recall__ or __sensitivity__. Selecting the appropriate model and metrics that demonstrate the aspects of a models performance are an import part of building interpretable models.

#### How can interpretability improve decision support?
Since predictive power is an important aspiration of a machine learning project, it is helpful to articulate what we hope to gain from our predictions. Below, we summarize the main themes in the relationship between machine learning goals and interpretability.

1. __Trust__ - performance metrics like accuracy, ROC/AUC, and F1 that allow us to evaluate our model's errors from various perspectives. __They let us know how many mistakes our model has made as well as the __nature of the mistakes__.


2. __Causality__ - Supervised models that help illustrate the associations between variables and outcomes can help researchers to generate hypotheses that can be tested to solve real world problems. __While correlation does not equal causation, the correlation between variables can nudge us in the right direction as what interventions we should try__. 


3. __Transferability__ - Model Interpretability contributes to our ability to assess if the model can still perform well if it is transferred to another scenario. Often, models will perform poorly when applied to new scenarios because they do not filter information in the same way as humans do. __If we do not understand the drivers of a decision made by our model, it is more challenging to debug or hypertune in order to improve performance__.  


4. __Informativeness__ - For a model to be useful and informative in a real world context, the model should output metrics that support the reasons for its decision. This allows researchers assess the importance of features they may not have considered, as well as the ranking of each features importance. __The model is allowing experts to enhance their understanding of a subject by providing novel information__.


5. __Fair and Ethical Decision Making__ - Since algorithms have come to impact our lives in nearly every aspect, there is a growing importance that the algorithms themselves do not exacerbate existing injustices in our society. Interpretability and explainability provides accountability for organizations using AI.

## White Box versus Black Box Models
In the past, when algorithms were less sophisticated and computational resources were limited the question of interpretability and explainability was less complex. Regression and simple tree based classifiers became tried and true options for predictive analytics. Well-understood metrics existed to interpret the models, which increased the confidence of stakeholders when using them to make decisions. While complex models like neural networks have existed since the 1980s, limitations in compute power and data infrastructure limited their use. As compute power increased and data infrastructure became more robust, the predictive power of neural networks was revealed making them a popular choice. 

When leveraged correctly, neural networks can deliver high quality results, however interpreting neural networks (and other complex algorithms) is not as straight forward as interpreting regression and simple tree based models, earning them the "black box" moniker. The metaphor of the black box is to evoke the idea that the mechanism that informs the model's decision is opaque. In contrast, regression and simple tree based models are seen as transparent.

It is due to this perceived opaqueness that neural networks and other black box models are inappropriate for certain use cases. However, this doesn't mean that black boxes are completely uninterpretable -- they just require different methods to analyze.

## Summary

Interpretability in machine learning can be slippery to define, but focusing on the goals of a machine learning project can help us understand what aspects of interpretability should factor into model selection. When applying machine learning to support human decision making, interpretability is an important consideration to ensure that the application actually improves the efficiency and quality of the human decisions by providing more relevant information. Model interpretability can help us to debug or improve machine learning applications by allowing us to better understand the mechanism behind the decision.

In the upcoming lessons, we will discuss white box and black box models in more detail and demonstrate how these models can be used and interpreted in various contexts. 
