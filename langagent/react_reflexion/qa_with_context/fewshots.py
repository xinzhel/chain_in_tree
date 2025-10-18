##### CoT Actor in Policy-Only Workflow (3 demos) #####
COT = """Relevant Context: The Nile River is the longest river in the world, spanning approximately 6,650 kilometers (4,132 miles) in length. It flows through eleven countries in northeastern Africa, including Egypt, Sudan, and Uganda.
Question: What is the longest river in the world?
Thought: The question asks for the longest river in the world, which I know is the Nile River based on the context provided.
Action: Finish[Nile River]

Relevant Context: Ludwig van Beethoven was a German composer and pianist who was a crucial figure in the transition between the Classical and Romantic eras in Western classical music. One of his most famous compositions is the Symphony No. 9, also known as the "Choral" symphony.
Question: Which composer created Symphony No. 9?
Thought: The question is asking for the composer of Symphony No. 9. Based on the context, I know that Ludwig van Beethoven composed this symphony.
Action: Finish[Ludwig van Beethoven]

Relevant Context: Photosynthesis is the process by which green plants and some other organisms convert light energy into chemical energy. During this process, plants absorb sunlight, carbon dioxide, and water to produce glucose and oxygen.
Question: What do plants produce during photosynthesis?
Thought: The question is asking about the products of photosynthesis. From the context, I know that plants produce glucose and oxygen during this process.
Action: Finish[Glucose and Oxygen]
"""

##### Evaluator in Policy-Only + FL Workflow (from Evaluator) (2 demos) #####
COT_REFLECT = """
Relevant Context: Ernest Hemingway's novel "The Old Man and the Sea" tells the story of Santiago, an aging Cuban fisherman, who struggles to catch a giant marlin in the Gulf Stream. The book won the Pulitzer Prize for Fiction in 1953 and contributed to Hemingway's Nobel Prize for Literature in 1954.
Question: Which literary award did "The Old Man and the Sea" contribute to Hemingway winning?
Thought: The question is asking which award "The Old Man and the Sea" contributed to Hemingway winning. Based on the context, I know the novel won the Pulitzer Prize for Fiction and contributed to his Nobel Prize for Literature.
Action: Finish[Pulitzer Prize for Fiction]

Reflection: My answer was correct based on the context, but may not be the exact answer stored by the grading environment. Next time, I should try to provide a less verbose answer like "Pulitzer Prize" or "Nobel Prize."

Context: On 14 October 1947, Chuck Yeager, a United States Air Force test pilot, became the first person to break the sound barrier by flying the Bell X-1 experimental aircraft at an altitude of 45,000 feet.
Charles Elwood "Chuck" Yeager (13 February 1923 - 7 December 2020) was a United States Air Force officer, flying ace, and test pilot. He is best known for becoming the first person to break the sound barrier, which he achieved in the Bell X-1 aircraft named Glamorous Glennis. Yeager was also a distinguished fighter pilot during World War II and was credited with shooting down at least 12 enemy aircraft. In 1973, he was inducted into the National Aviation Hall of Fame for his significant contributions to aviation.
Question: Who is the first person to break the sound barrier?
Thought: The question is asking for the first person to break the sound barrier. From the context, I know that Chuck Yeager, a United States Air Force test pilot, was the first person to break the sound barrier.
Action: Finish[Chuck Yeager]

Reflection: Upon reflecting on the incorrect answer I provided, I realize that I may not have given the full name of the individual in question. In the context, both the given name and the nickname were mentioned, and I only used the nickname in my response. This could have been the reason my answer was deemed incorrect. Moving forward, when attempting this question again or similar questions, I will make sure to include the complete name of the person, which consists of their given name, any middle names, and their nickname (if applicable). This will help ensure that my answer is more accurate and comprehensive."""

##### Evaluator in Policy-Only + FL Workflow (from Evaluator) (3 demos) #####
COT_REFLECT2 = """Relevant Context: The novel "To Kill a Mockingbird" was written by Harper Lee and published in 1960. The story takes place in the fictional town of Maycomb, Alabama during the Great Depression. The main characters are Scout Finch, her brother Jem, and their father Atticus Finch, a lawyer.
Question: Where does "To Kill a Mockingbird" take place?
Thought: The question is asking for the setting of "To Kill a Mockingbird." Based on the context, I know that the story takes place in the fictional town of Maycomb, Alabama.
Action: Finish[Mississippi]

Reflection: I made an error in my response, as I incorrectly stated that "To Kill a Mockingbird" takes place in Mississippi. Upon reviewing the context, I realized that the correct answer is the fictional town of Maycomb, Alabama. I may have been confused due to the story's Southern setting. Next time, I should be more cautious and double-check the context before providing an answer.

Relevant Context: Sir Isaac Newton formulated the three laws of motion that are fundamental to classical mechanics. These laws describe the relationship between the motion of an object and the forces acting upon it. They are known as Newton's First Law, Newton's Second Law, and Newton's Third Law.
Question: Who formulated the three laws of motion?
Thought: The question is asking for the person who formulated the three laws of motion. Based on the context, I know that Sir Isaac Newton formulated these laws.
Action: Finish[Albert Einstein]

Reflection: My response was incorrect, as I stated that Albert Einstein formulated the three laws of motion, when the correct answer is Sir Isaac Newton. I may have confused the two scientists due to their significant contributions to the field of physics. In the future, I should be more careful in distinguishing between different scientists and their specific accomplishments before answering a question.

Relevant Context: The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials. It was built primarily to protect the Chinese Empire from invasions by various nomadic groups. The wall stretches over 13,000 miles (21,196 kilometers) and was constructed by several Chinese dynasties.
Question: What is the primary purpose of the Great Wall of China?
Thought: The question is asking about the primary purpose of the Great Wall of China. From the context, I know that it was built primarily to protect the Chinese Empire from invasions by various nomadic groups.
Action: Finish[Trade]

Reflection: My answer was incorrect, as I stated that the primary purpose of the Great Wall of China was for trade. After reevaluating the context, I realized that the wall was built primarily to protect the Chinese Empire from invasions by various nomadic groups. My confusion may have stemmed from the fact that the wall is a significant historical landmark that may have played a role in trade routes. Next time, I will focus on the key details provided in the context to ensure the accuracy of my response.
"""