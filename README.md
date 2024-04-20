<div align="center">
  <h2><b>Portfolio Projects<b></h2>
</div>

&nbsp;

<details>
  <summary><b>Click Here To Navigate To Each Repository<b></summary>

  - [Tackle Opportunity Window](https://github.com/TaberNater96/Portfolio-Projects/tree/main/NFL%20Big%20Data%20Bowl%202024)
</details>

&nbsp;

This GitHub repository features a diverse collection of projects, highlighting my contributions to sports analytics and environmental conservation through data science. A standout project is my participation in the NFL Big Data Bowl 2024, where I pioneered the Tackle Opportunity Window (TOW) metric. Leveraging TOW, along with other detailed metrics provided by the NFL on a frame-by-frame basis, I developed a neural network model that predicts the likelihood of a defensive player making a tackle. This metric evaluates the effectiveness of defensive players in making tackles by considering their physical and tactical skills. It has been well-received in the analytics community and earned recognition in the Kaggle competition from my data science peers.

Additionally, I am currently spearheading a project aimed at giving detailed insight into our planet's greenhouse gas emissions. This project encompasses the collection of data via NOAA's API using a custom-developed algorithm, followed by a rigorous process of cleaning, transforming, and loading the data into an AWS database. Subsequently, I will undertake exploratory data analysis and feature engineering to finely tune the data for a specially designed deep neural network built from scratch aimed at forecasting emission trends. The goal is to create a Streamlit dashboard that clearly presents emissions data from the past, present, and future to provide environmental insights to all as there are plenty of sources on the web for getting raw data on greenhouse gas emissions, but hardly any visualizations that provide insight into past, present, and future trends. This project will be done and published within the next 2-3 months. 

<div align="center">
  <h2>Overviews (TLDR)</h2>
</div>

## Table of Contents
- [Tackle Opportunity Window](#tackle-opportunity-window)

<div id="tackle-opportunity-window" align="center">
  <h2>Tackle Opportunity Window</h2>
</div>

My portfolio project, "Tackle Opportunity Window (TOW)," presented at the NFL Big Data Bowl 2024, introduces a groundbreaking metric to analyze and predict the defensive play in football. The TOW metric quantifies the crucial timeframe a defender has to execute a tackle, showcasing my capability to handle and extract meaningful insights from extensive and complex datasets containing over 12 million rows of high dimensional data.

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

To speak plainly, I am indeed the first to develop the TOW metric within the field of football analytics, demonstrating an unparalleled ability to innovate and pioneer new methodologies in sports analytics. This metric offers a fresh perspective by focusing on the precise moments a player remains within a strategic distance to make a successful tackle, thereby introducing a novel way to evaluate defensive actions. The project not only showcases my technical skills in handling large-scale data and complex statistical models but also highlights my creativity in generating new solutions to analyze performance metrics. By identifying and crafting a new evaluative metric, I have set a new standard in sports analytics for projects to build off of when analyzing tackle metrics, proving my capacity to lead and innovate in the field. This initiative demonstrates not just analytical acumen but also a visionary approach to transforming how data insights drive strategic decisions.
























