<div align="center">
  <h2>Overviews (TLDR)</h2>
</div>

## Table of Contents
- [Tackle Opportunity Window](#tackle-opportunity-window)
- [Climate Insight](#climate-insight)

<div id="tackle-opportunity-window" align="center">
  <h2>Tackle Opportunity Window</h2>
</div>

<a href="https://www.kaggle.com/code/godragons6/tackle-opportunity-window" target="_blank"><img align="left" alt="Kaggle" title="View Competition Submission" src="https://kaggle.com/static/images/open-in-kaggle.svg"></a>

&nbsp;

My portfolio project, "Tackle Opportunity Window (TOW)," presented at the NFL Big Data Bowl 2024, introduces a novel metric to analyze and predict the defensive play in football. The TOW metric quantifies the crucial timeframe a defender has to execute a tackle, showcasing my capability to handle and extract meaningful insights from extensive and complex datasets containing over 12 million rows of high dimensional data. This competition was one of my first data science projects where python and machine learning were very new concepts to me. I actually joined this competition when it was over half way done, but I decided to fully dive in, working through the holidays, and did not stop until the final model, simulation, and report was complete (refer to the NFL repository README for full report). This competition repo differs from standard data science projects since the entire codebase is in a single notebook. This is so the notebook can be uploaded to Kaggle and viewed by other Data Scientists.

The neural network I developed for this project processes real-time player tracking data to predict tackle probabilities, a key output visualized using Plotly Express that I built from scratch. As the game progresses, each player's position relative to the ball carrier is fed into the model, which calculates the likelihood of a tackle. This probability is graphically represented by a dynamic 'probability bubble' around each player on the field. These bubbles grow in size as players enter closer proximity to the ball carrier, aligning with the increasing Tackle Opportunity Window (TOW). This visualization technique not only illustrates the model’s predictive capabilities but also provides an intuitive display of shifting tackle probabilities during live gameplay, highlighting the application of advanced neural network analysis in real-time sports scenarios.

<div align="center">
<img src="https://github.com/TaberNater96/Portfolio-Projects/blob/main/NFL%20Big%20Data%20Bowl%202024/Images/Players/Marshon%20and%20Tyrann.png?raw=true" width="800" height="90">
</div>

<div align="center">
<img src="https://github.com/TaberNater96/Portfolio-Projects/blob/main/NFL%20Big%20Data%20Bowl%202024/Images/TOW%20Animation.gif?raw=true" width="800" height="400">
</div>

In developing TOW, I leveraged a sophisticated approach to manage the sheer volume of data efficiently. The methodology involved calculating the dynamic Euclidean distance between a defender and the ball carrier across multiple frames, utilizing a vectorized computation method. This innovation not only optimized the processing of vast datasets but also illuminated hidden patterns within the chaotic and fast-paced movements of NFL games. The result is a unique, finely tuned metric that significantly enhances our understanding of defensive tactics and player effectiveness.

<div align="center">
<img src="https://github.com/TaberNater96/Portfolio-Projects/blob/main/NFL%20Big%20Data%20Bowl%202024/Images/Players/Marshon%20and%20Tyrann.png?raw=true" width="800" height="90">
</div>

<div align="center">
<img src="https://github.com/TaberNater96/Portfolio-Projects/blob/main/NFL%20Big%20Data%20Bowl%202024/Images/TOW%20Plot%20Animation.gif?raw=true" width="800" height="400">
</div>

Owing to the intrinsic capabilities of a neural network, particularly its adeptness in detecting nuanced variations, the direction and orientation of each player were pivotal in enabling the model to discern and adapt to subtle shifts. These shifts are essential for the model to recognize and adhere to an emergent pattern, as the features exhibit significant, yet controlled, variation across successive frames. This variation is not arbitrary, but rather demonstrative of a tackler pursuing their target with precision. The controlled variability within these features provides the model with critical data points, allowing it to effectively learn and predict the dynamics of a tackler's movement in relation to their target. Visualizing the distribution of each player's orientation and direction in the EDA phase, and noticing the non-random variation, is what gave rise to the idea of focusing on this specific concept in parallel with the tackle opportunity window. 

<div align="center">
<img src="https://github.com/TaberNater96/Portfolio-Projects/blob/main/NFL%20Big%20Data%20Bowl%202024/Images/Polar%20Histogram.png?raw=true" width="600" height="600">
</div>

To speak plainly, I am indeed the first to develop the TOW metric within the field of football analytics, demonstrating an ability to innovate and pioneer new methodologies in the realm of data science. This metric offers a fresh perspective by focusing on the precise moments a player remains within a strategic distance to make a successful tackle, thereby introducing a novel way to evaluate defensive actions. The project not only showcases my technical skills in handling large-scale data and complex statistical models but also highlights my creativity in generating new solutions to analyze performance metrics. By identifying and crafting a new evaluative metric, I have set a new standard in sports analytics for projects to build off of when analyzing tackle metrics, proving my capacity to lead and innovate in the field. This initiative demonstrates not just analytical acumen but also a visionary approach to transforming how data insights drive strategic decisions.

<div id="climate-insight" align="center">
  <h2>Climate Insight</h2>
</div>

## This project is currently under development.

Climate Insight is a full stack NLP pipeline project, whereby I train multiple advanced deep learning models to classify and summarize complex climate change articles to derive meaningful insights. This project is 6 months in and will be done within 3-4 months. Progress so far:

- Developed a custom web crawler and web scraper to find and scrape open source climate change research articles for training.
- Crafted the architecture to train and evaluate the following deep learning NLP models:
    - **CLIMATBart**: Hybrid Text Summarization Engine using a custom extractive model and abstractive BART model.
    - **RoBi**: Hybrid Sentiment Analysis Network using RoBERTa and BiLSTM.
    - **CLIMATopic**: Topic model using BERTopic.
    - **C4NN**: Text classification model using a convolutional neural network built from scratch. 
- Currently developing the front end website and backend server.





















