# Agent Strategy for ANAC 2024

## By: Omer Bartfeld, Eilon Alter, Aviv Nataf

In preparation for this year's ANAC competition, our team embarked on a journey of meticulous planning and execution to craft an agent strategy that not only meets but exceeds the challenges posed. With an unwavering commitment to innovation and excellence, we endeavored to develop a dynamic negotiation framework that harnesses the power of real-time learning and adaptive decision-making. This strategy aims to equip our agent with the ability to discern and respond to the nuanced behaviors of its counterparts, leveraging historical insights to drive optimal concession strategies.

## Comprehensive Strategy Development

Our strategy development process was marked by intensive brainstorming sessions and collaborative efforts aimed at conceptualizing a holistic approach to negotiation. Recognizing the pivotal role of understanding our opponents' preferences and constraints, we placed a strong emphasis on devising a strategy that dynamically adapts to evolving negotiation dynamics. Through meticulous analysis and strategic foresight, we sought to create a framework that not only responds to known variables but also thrives in the face of uncertainty.

## Key Considerations: Utility Function and Deadline Time

Central to our strategy formulation was a deep understanding of the opponent's utility function and deadline time. These fundamental parameters serve as the cornerstone of negotiation dynamics, guiding our agent's decision-making process. By gaining insights into the underlying motivations and constraints of our counterparts, we aimed to tailor our negotiation approach to maximize mutual gains while safeguarding our interests.

## The Missing Puzzle Piece: Reservation Value

One of the primary challenges we encountered in crafting our strategy was the absence of crucial information regarding the opponent's reservation value. This elusive parameter represents the threshold beyond which concessions become untenable, making it a critical determinant of negotiation outcomes. Recognizing the significance of this missing puzzle piece, our strategy pivoted towards a dynamic learning approach aimed at unraveling the opponent's reservation value over the course of negotiations.

## Innovative Bilateral Negotiation Model

Aligned with the rules and conditions set forth by ANL, we proposed an innovative bilateral negotiation model grounded in Bayesian learning principles. This model empowers our agent to adapt its negotiation strategies dynamically, leveraging real-time feedback to optimize outcomes. By embracing a time-dependent tactical approach, we aimed to foster an environment conducive to mutual understanding and cooperation.

## Navigating the Negotiation Landscape

Our negotiation process unfolds through a series of meticulously crafted steps, each designed to unravel the complexities of the bargaining landscape:

1. **Initialization and Detection Region Setting**: We kickstart the negotiation process by establishing a detection region to estimate the opponent's reservation value. This region is meticulously divided into cells, with each cell representing a potential range of reservation values. Drawing upon prior knowledge and assumptions, we assign initial beliefs to each cell, laying the groundwork for subsequent analysis.

2. **Regression Analysis**: At each iteration of the negotiation, we employ regression analysis to generate insights into the opponent's behavior. By simulating random reservation points within each cell, we construct regression curves that capture the relationship between concession rates and potential reservation values. This analysis provides invaluable insights into the opponent's decision-making process, enabling us to adapt our strategies accordingly.

3. **Bayesian Learning Integration**: Leveraging Bayesian learning principles, we integrate historical offers and regression analysis results to refine our understanding of the opponent's preferences. By updating our beliefs iteratively, we enhance the accuracy of our predictions and tailor our concession strategies to maximize utility.

4. **Concession Point Calculation**: Armed with probabilistic insights, we proceed to calculate concession points that optimize mutual gains. By identifying common ground between our rational outcomes and the opponent's preferences, we strive to foster a collaborative negotiation environment conducive to reaching favorable agreements.

5. **Concession Value Determination**: Finally, we translate probabilistic insights into actionable concession strategies, dynamically adjusting our offers based on evolving negotiation dynamics. By harnessing the power of real-time learning, we ensure that our agent remains agile and responsive throughout the negotiation process, maximizing the likelihood of favorable outcomes.

## Conclusion

In conclusion, our refined agent strategy represents a culmination of innovative thinking, strategic foresight, and meticulous execution. By embracing a dynamic learning approach grounded in Bayesian principles, we equip our agent with the tools and insights necessary to navigate the complexities of negotiation with confidence and agility. As we embark on the ANAC 2024 competition, we remain steadfast in our commitment to excellence, poised to demonstrate the efficacy of our strategy on the global stage.
