# Awesome RLAIF
A continually updated list of literature on **Reinforcement Learning from AI Feedback (RLAIF)**.

## What is RLAIF? ChatGPT: 
Reinforcement Learning from AI Feedback (RLAIF) is a concept that describes a type of machine learning approach where an AI agent learns by receiving feedback or guidance from another AI system. This concept is closely related to the field of Reinforcement Learning (RL), which is a type of machine learning where an agent learns to make a sequence of decisions in an environment to maximize a cumulative reward.

In traditional RL, an agent interacts with an environment and receives feedback in the form of rewards or penalties based on the actions it takes. It learns to improve its decision-making over time to achieve its goals. In the context of Reinforcement Learning from AI Feedback, the AI agent still aims to learn optimal behavior through interactions, but the feedback comes from another AI system rather than from the environment or human evaluators. This can be particularly useful in situations where it may be challenging to define clear reward functions or when it is more efficient to use another AI system to provide guidance. The feedback from the AI system can take various forms, such as:
- Demonstrations: The AI system provides demonstrations of desired behavior, and the learning agent tries to imitate these demonstrations.
- Comparison Data: The AI system ranks or compares different actions taken by the learning agent, helping it to understand which actions are better or worse.
- Reward Shaping: The AI system provides additional reward signals to guide the learning agent's behavior, supplementing the rewards from the environment.

This approach is often used in scenarios where the RL agent needs to learn from limited human or expert feedback or when the reward signal from the environment is sparse or unclear. It can also be used to accelerate the learning process and make RL more sample-efficient. Reinforcement Learning from AI Feedback is an area of ongoing research and has applications in various domains, including robotics, autonomous vehicles, and game playing, among others.

## Papers
### 2023
- [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267)
  - Authors: Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, Abhinav Rastogi
  - <details> <summary>Abstract (click me)</summary> Reinforcement learning from human feedback (RLHF) is effective at aligning large language models (LLMs) to human preferences, but gathering high quality human preference labels is a key bottleneck. We conduct a head-to-head comparison of RLHF vs. RL from AI Feedback (RLAIF) - a technique where preferences are labeled by an off-the-shelf LLM in lieu of humans, and we find that they result in similar improvements. On the task of summarization, human evaluators prefer generations from both RLAIF and RLHF over a baseline supervised fine-tuned model in ~70% of cases. Furthermore, when asked to rate RLAIF vs. RLHF summaries, humans prefer both at equal rates. These results suggest that RLAIF can yield human-level performance, offering a potential solution to the scalability limitations of RLHF. </details>

### 2022
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
  - Authors: Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, et al.
  - <details> <summary>Abstract (click me)</summary> As AI systems become more capable, we would like to enlist their help to supervise other AIs. We experiment with methods for training a harmless AI assistant through self-improvement, without any human labels identifying harmful outputs. The only human oversight is provided through a list of rules or principles, and so we refer to the method as 'Constitutional AI'. The process involves both a supervised learning and a reinforcement learning phase. In the supervised phase we sample from an initial model, then generate self-critiques and revisions, and then finetune the original model on revised responses. In the RL phase, we sample from the finetuned model, use a model to evaluate which of the two samples is better, and then train a preference model from this dataset of AI preferences. We then train with RL using the preference model as the reward signal, i.e. we use 'RL from AI Feedback' (RLAIF). As a result we are able to train a harmless but non-evasive AI assistant that engages with harmful queries by explaining its objections to them. Both the SL and RL methods can leverage chain-of-thought style reasoning to improve the human-judged performance and transparency of AI decision making. These methods make it possible to control AI behavior more precisely and with far fewer human labels.  </details>
  - Links: [dataset](https://github.com/anthropics/ConstitutionalHarmlessnessPaper)

## Related Awesome - $\star$ repos
- [awesome-RLHF](https://github.com/opendilab/awesome-RLHF/tree/main)

## Contributing
Let's make the list more comprehensive. 



