<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/kenanGonnot/topic_modeling">

<h3 align="center">Topic Modeling</h3>

  <p align="center">
   By Kenan
    <br />
    <br />
    <br />
  </p>




<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul> </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## **_PS: I recommend you to use google collab to run the code._** 

## Run the app
```bash 
docker build -t thekenken/topic-modeling-demo:latest . && docker push thekenken/topic-modeling-demo:latest && docker run -p 5003:5000 thekenken/topic-modeling-demo:latest
```
for M1 Macs:
```bash
docker buildx build --platform linux/amd64 -t thekenken/topic-modeling-demo:latest . && docker push thekenken/topic-modeling-demo:latest
```

for dev:
```bash 
docker build -t thekenken/topic-modeling-demo:dev . && docker run -p 5003:5000 thekenken/topic-modeling-demo:dev
```

## About The Project


**GOAL :** To create a topic modeling project. 


There are two methods:
* using LDA model (lda_TopicModeling.ipynb)
* using a pipeline like this (pipeline_TopicModeling.ipynb):
![Image of pipeline](images/pipeline.png)

***

**DATASETS**: Here I scrapped some text in the web to create a dataset (this data is used for non-commercial purposes):   

| Name              | Qty document | 
|-------------------|--------------|
| French revolution | 9            |
| One piece         | 3            |
| Ancient Egypt     | 4            |
| Machine Learning  | 3            |
| **Python**        | **9**        | 
Python : **This is the topic I used** 

***
<!-- CONTACT -->

## Contact

Student: 
* Kenan Gonnot - [linkedin](https://www.linkedin.com/in/kenan-gonnot/) 

