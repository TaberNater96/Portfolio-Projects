<center><h1>Tackle Opportunity Window</h1></center>
&nbsp;

#### Analysis & Scripting By: Elijah Taber
#### Submission Track: Undergraduate Track

# Introduction

Imagine being a running back in the NFL and having a 6'4", 250 lb defensive linebacker, with a 4.39 40 yard dash, rapidly approaching you and tracking your every position. As an observer watching from above, it's hard to gain a solid understanding for what is actully happening down on the field, and what is going through the minds of the players in such a high stakes situation. Which player do you think will come out on top? Is it the player who is faster? Stronger? Smarter? A more nuanced analysis suggests that it is an amalgamation of these characteristics that defines the prowess of a professional football player. The main objective when the ball is snapped for every defensive player is to stop the ball carrier as soon as possible. Hence, the pivotal question arises: what are the paramount factors that facilitate this objective? 

To address this, I have developed a unique metric, termed the **Tackle Opportunity Window**, or **TOW**. This metric quantifies the duration within which a defensive player can feasibly execute a tackle. The NFL contains some of the fastest and strongest human beings on the planet. On many occasions in football, a player on defense only has a fraction of a second to bring down a ball carrier before they are out of reach and must be brought down by a different defensive player. Therefore, an effective metric to assess a defender's tackling proficiency is to examine their actions during the critical moments preceding a tackle. A robust TOW score is indicative of a player's capacity to adeptly track, rapidly accelerate, and maintain a strategic proximity to the ball carrier. This proficiency is derived from an interplay of experience, skill, and physical capabilities, each contributing to the player's overall effectiveness on the field.

# Theory

The functionality of this metric stems from calculating a dynamic variant of the Euclidean Distance across successive frames in order to determine the distance of the defensive players with respect to the ball carrier over each frame. This calculation is pivotal in ascertaining the fluctuating proximity of defensive players to the ball carrier within each frame. The variation in this distance, for each player, is contingent upon their relative Cartesian coordinates mapped onto the two-dimensional plane of the football field. 

#### Relative Distance
The Euclidean distance \( d \) between a player and the ball carrier is calculated as:

$$
d = \sqrt{(x_{\text{player}} - x_{\text{ball carrier}})^2 + (y_{\text{player}} - y_{\text{ball carrier}})^2}
$$

#### Threshold
Once the relative distance of each defensive player with respect to the ball carrier is known, it can be compared to the TOW threshold, which I have set to 2 yards. This threshold value can be adjusted if need be. If the distance is less than the threshold, the TOW counter will start. 

$$
wt = d \leq \theta
$$

Where:

$$
\theta = \text{threshold},
$$

$$
wt = \text{within tackle threshold}
$$

#### Tackle Opportunity Window
For each consecutive frame that the defensive player remains within that threshold for the duration of the play, the counter will add a score of 1. Once the play is over, the algorithm will check for the largest number and assign that score to each player. 

$$
TOW (\theta) = \max_{\text{score}}\left(\sum_{i=1}^{n} wt_i - \min_{j \leq i}\left(\sum_{k=1}^{j} wt_k \cdot (1 - wt_j)\right)\right)
$$

Due to the scale of this operation, transforming all of weeks (1 – 9) by looping over each row would be highly inefficient. Therefore I implemented a vectorized approach rather than deep nested loops to optimize the iteration process. Once the window is verified, the TOW counter is instantiated to create a cumulative score. This can be visualized using the following: 

  $$
  \text {TOW Score}_i \text{+=} 
  \begin{cases} 
  1 & \text{if } d_i \leq \theta \\
  0 & \text{otherwise}
  \end{cases}
  $$
  
#### Tackle Opportunity Window Ratio
The TOW score is a valuable metric for comparing players within an individual play. However, its effectiveness diminishes when assessing performances across multiple tackles over a nine-week period. This is because longer plays naturally lead to higher TOW scores, regardless of the actual time a tackler spends near the ball carrier. To address this, introducing a ratio-based system would offer a more accurate and fair representation of each player's performance in every distinct play.

$$
\text{TOW Ratio} = {\text{TOW Score} \over \text{Total Frames Per Play}} 
$$

By leveraging geometric principles, the TOW metric provides a nuanced understanding of the spatial dynamics that influence and predict tackling outcomes in the game. To gain some insight into the overall distribution of the TOW ratio for the entire tracking dataset, I crafted a 2D density plot of tacklers with a non-zero TOW Ratio to illuminate how consistent tacklers are at maintaining a close proximity to their target. 

<div align="center">
<img src="https://github.com/TaberNater96/NFL-Big-Data-Bowl-2024/blob/main/Images/TOW%20Ratio%20Density%20Plot.png?raw=true" width="600" height="400">
    </div>

As one can see, the majority of tacklers stay within the tackle threshold for around 1/3 of the time, where the majority of the score is most likely at the end of the play.

# Data Preprocessing

#### Cross Mapping

In order to create a binary classification target variable that is aptly termed ‘tackle’, the dataset must be synthesized by extracting and amalgamating pertinent metrics from a plethora of distinct dataframes. This would be straight forward if the tracking datasets set one tackle value in the ‘event’ column, but the same events are recorded across each player. To circumvent this issue, and engineer a target variable, I cross referenced the tackles dataframe with my main tracking dataframe to map a binary indicator of 1 on the exact frame where a tackle occurred, but only on the specific nflId(s) that is responsible for the tackle. At this a point the tackler(s), ball carrier, and relative spatial separation from the two are known, allowing for rudimentary trigonometry computations to determine if the threshold has been passed, initiating the TOW counter. 

#### Data Normalization and Transformation

The tracking datasets underpinning this analysis is extensive, containing 9 CSV files that collectively encompass over 12 million data entries. A critical aspect of this dataset is the 'gameId' attribute, which serves as a distinct identifier for each game. However, the 'playId' attribute, while unique within the context of a single game, is not globally unique across different games. This implies that while each game is associated with a distinct set of 'playId' numbers, these identifiers are recycled across different games. This makes things difficult when trying to group the playId's by their unique values in order to filter out a tackle without combining mutiple separate plays into one due to their ID's being the same value. To address this complexity, a transformation was implemented by numerically encoding 'playId' values, culminating in the creation of a new, distinct attribute. This attribute iterates through the entire dataset, assigning a unique value to each play. This process ensures the preservation of the dataset's structural integrity and chronological coherence. 

A good way to visualize the data phenomena is by looking at the tackles dataframe, sorted by playId. Notice recycled values in the playId column:

<div align="center">
<img src="https://github.com/TaberNater96/NFL-Big-Data-Bowl-2024/blob/main/Images/Tackles%20Dataframe%20Example.png?raw=true" width="600" height="400">
    </div>

# Deep Neural Network

#### Predictive Tracking Indicators
Owing to the intrinsic capabilities of a neural network, particularly its adeptness in detecting nuanced variations, the direction and orientation of each player were pivotal in enabling the model to discern and adapt to subtle shifts. These shifts are essential for the model to recognize and adhere to an emergent pattern, as the features exhibit significant, yet controlled, variation across successive frames. This variation is not arbitrary, but rather demonstrative of a tackler pursuing their target with precision. The controlled variability within these features provides the model with critical data points, allowing it to effectively learn and predict the dynamics of a tackler's movement in relation to their target. Visualizing the distribution of each player's orientation and direction in the EDA phase, and noticing the non-random variation, is what gave rise to the idea of focusing on this specific concept in parallel with the tackle opportunity window. 

<div align="center">
<img src="https://github.com/TaberNater96/NFL-Big-Data-Bowl-2024/blob/main/Images/Polar%20Histogram.png?raw=true" width="600" height="600">
</div>

#### Architecture
When the target variable is split, its training set is used to compute the class weight of the network using balanced parameters to address the disproportionality of the tackle ratio. The training and test sets are then split into categorical and continuous variables, then embedded into the network to capture underlying connections by mapping integers to dense vectors. The network is then compiled using the loss function as binary-crossentropy, with a focus on tracking performance. Since overfitting is a common issue in neural networks, the features were kept to a necessary minimum. The Adam algorithm uses a tunable learning rate to ensure this hyperparameter is set to an optimal frequency. The architecture framework I implemented analyzes the validation loss of each epoch and then automatically extracts the highest performing parameters. Validation loss is favored over validation accuracy due to the nature of the binary output layer and the overall balance the target variable.

# Results

#### Tackle Opportunity Window & Orientation In Context

To provide a deeper understanding of how this model operates in practice, consider the following play as an illustrative example. The yellow radius around the ball illustrates the TOW threshold while the black radius around the defenders represents their respective tackle probability density. Notice how the densities for Tyrann Mathieu and Marshon Lattimore fluctuate, influenced by their orientation and distance to the ball carrier. These two factors are critical. As soon as the TOW threshold is breached, their probability densities spike. 

<div align="center">
<img src="https://github.com/TaberNater96/NFL-Big-Data-Bowl-2024/blob/main/Images/Players/Marshon%20and%20Tyrann.png?raw=true" width="800" height="100">
</div>

<div align="center">
<img src="https://github.com/TaberNater96/NFL-Big-Data-Bowl-2024/blob/main/Images/TOW%20Animation.gif?raw=true" width="800" height="400">
</div>

#### Time Series Analysis
Defenders who can maintain a consistent tackle opportunity window by adjusting their orientation and acceleration on a moment’s notice can double or triple their chances of securing a tackle. This is due to the fast-paced environment of the NFL and how split second decisions determine the outcome of a play. The following graph illustrates this concept by analyzing how Tyrann Mathieu and Marshon Lattimore's probabilities fluctuate as their orientation and distance to the ball carrier change.

<div align="center">
<img src="https://github.com/TaberNater96/NFL-Big-Data-Bowl-2024/blob/main/Images/Players/Marshon%20and%20Tyrann.png?raw=true" width="800" height="100">
</div>

<div align="center">
<img src="https://github.com/TaberNater96/NFL-Big-Data-Bowl-2024/blob/main/Images/TOW%20Plot%20Animation.gif?raw=true" width="800" height="400">
</div>

#### Angle of Pursuit
The following play by Yetur Gross-Matos, an outside linebacker for the Carolina Panthers, illustrates how an angle of pursuit is invaluable in tackling a ball carrier that is already on the run at full speed. His ability to track, accelerate, and properly angle himself, resulting in a crucial tackle, exemplifies his effort in not giving up on the play.

<div align="center">
<img src="https://github.com/TaberNater96/NFL-Big-Data-Bowl-2024/blob/main/Images/Players/Gross-Matos.png?raw=true" width="600" height="100">
</div>

<div align="center">
<img src="https://github.com/TaberNater96/NFL-Big-Data-Bowl-2024/blob/main/Images/Angle%20of%20Pursuit.gif?raw=true" width="800" height="400">
</div>

# Rankings

The tackle opportunity window excels at demonstrating how efficient defenders are at tracking and maintaining a consistent proximity to their respective target. This parameter, however, necessitates contextual interpretation, as the positional roles of players inherently present divergent scopes for target tracking, contingent upon their spatial deployment and the dynamic opposition presented by offensive counterparts. This results in some players receiving more resistance than others. Therefore, I have developed a tripartite ranking system, categorizing defensive players' tracking capabilities in accordance with their positional classifications.

<div align="center">
<img src="https://github.com/TaberNater96/NFL-Big-Data-Bowl-2024/blob/main/Images/DL%20Rankings.png?raw=true" width="800" height="600">
    </div>

<div align="center">
<img src="https://github.com/TaberNater96/NFL-Big-Data-Bowl-2024/blob/main/Images/Linebacker%20Rankings.png?raw=true" width="800" height="600">
    </div>

<div align="center">
<img src="https://github.com/TaberNater96/NFL-Big-Data-Bowl-2024/blob/main/Images/DB%20Rankings.png?raw=true" width="800" height="600">
    </div>

# Conclusion

The deep learning network I implemented demonstrated solid proficiency in identifying not just the instances where defensive players are narrowing the gap with the ball carrier, but also in recognizing their alignment and predicting their optimal positioning for successful tackles. It was observed that mere proximity to the ball carrier is insufficient for a high probability of tackling; proper orientation and strategic positioning are crucial. This insight is particularly relevant in the NFL, where the combination of speed, strength, and agility often overshadows the necessity for tactical placement and orientation, especially when players are outside the immediate tackle threshold.

The introduction of the 'tackle opportunity window' metric marks a significant advancement in evaluating defensive player performance. This metric offers a novel perspective by illustrating how effectively individual players track, accelerate towards, and maintain strategic proximity to the ball carrier. It also highlights their ability to anticipate and align themselves accurately with the future movements of the ball carrier. This interplay is a synthesis of rigorous training, experience, and overall intuition that characterizes a professional football player. Overall, this study provides valuable insights into the complex dynamics of defensive play in the NFL, underscoring the intricate balance between physical prowess and tactical intelligence.

Word Count: 1927

Figures: 9
