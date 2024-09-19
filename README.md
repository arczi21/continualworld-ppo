# On-Policy Algorithms for Continual Reinforcement Learning

## Abstract

Continual reinforcement learning (CRL) is the study of optimal strategies for maximizing rewards in sequential environments that change over time. This is particularly crucial in domains such as robotics, where the operational environment is inherently dynamic and subject to continual change. Nevertheless, research in this area has thus far concentrated on off-policy algorithms with replay buffers that are capable of amortizing the impact of distribution shifts. Such an approach is not feasible with on-policy reinforcement learning algorithms that learn solely from the data obtained from the current policy. In this paper, we examine the performance of proximal policy optimization (PPO), a prevalent on-policy reinforcement learning (RL) algorithm, in a classical CRL benchmark. Our findings suggest that the current methods are suboptimal in terms of average performance. Nevertheless, they demonstrate encouraging competitive outcomes with respect to forward transfer and forgetting metrics. This highlights the need for further research into continual on-policy reinforcement learning.

## Introduction

The assumption of numerous machine learning algorithms that data follows an independent and identically distributed (i.i.d.) pattern is frequently invalid, given that the real world is in a constant state of flux. Such a shift may occur in the observed data distribution during the training phase, yet it is not accounted for by the aforementioned algorithms. Continual learning (CL) represents a field of study that aims to address these issues. However, its application for solving reinforcement learning (RL) tasks presents considerably greater challenges than those encountered in the context of simple classification problems. The Continual World robotic benchmark, as proposed by [Wołczyk et al. (2021)](https://arxiv.org/abs/2105.10919 "Wołczyk et al. (2021)"), is a specific tool designed for the evaluation of RL agents in CL environments. The primary limitation of this tool is that it relies solely on the off-policy soft actor-critic (SAC) algorithm [Haarnoja et al. 2018](https://arxiv.org/abs/1801.01290 "Haarnoja et al. 2018"). The aim of this study is to investigate whether existing approaches to continual learning can be effectively combined with proximal policy optimization (PPO), a widely used on-policy algorithm ([Schulman et al. 2017](https://arxiv.org/abs/1707.06347 "Schulman et al. 2017")). Our findings indicate that, while the use of PPO in lieu of SAC typically results in markedly inferior average performance, this is not the case with regard to forward transfer and forgetting metrics. We suspect that this phenomenon may be attributed to the exploitation of off-policy algorithms from a replay buffer, which provides access to data generated by any policy. The obtained results suggest the necessity for further investigation into on-policy continual reinforcement learning (CRL). In subsequent work, we intend to develop CL methods that will address the identified shortcomings.

## RL and CRL in Brief

Reinforcement learning is a framework for modeling and solving decision-making problems where an agent interacts with a dynamic environment to maximize a cumulative reward over time. In contrast, continual reinforcement learning is a domain where the agent encounters a sequence of tasks (with different environments) rather than a single, isolated one. This setup presents two principal challenges: catastrophic forgetting, whereby the agent loses the ability to perform previous tasks after learning new ones, and transfer learning, which entails applying knowledge from one task to enhance learning in related new tasks. Accordingly, in lieu of an average *performance* (success rate) across all tasks, a more comprehensive evaluation of CRL necessitates the utilization of sophisticated metrics such as average *forward transfer* (normalized area between training curves of the task in a sequence and detached) and average *forgetting* (difference between success rate on the task at the end of its training and at the end of the entire learning process).

##### Average performance

Let $p_i(t)$ denote the success rate of the model on task $i$ at time $t$, which is defined as the average number of times the agent successfully reaches the goal of task $i$ during the evaluation phase with stochastic policy at time $t$. The average performance is then defined as the mean success rate across all $N$ tasks at a final evaluation time $T$:
<p align="center">
$\text{AP} = \frac{1}{N} \sum_{i=1}^{N} p_{i}(T)$.
</p>

##### Forward transfer

We measure the forward transfer of a task as a normalized area between the training curve of a particular task (trained in a sequence) and the training curve of the reference (a single-task experiment). Let $p_{i}^{b}(t) \in\left[0,1\right]$ be the success rate of the model on the reference of task $i$ and $p_i(t)$ be the success rate of the model on task $i$ at time $t$. The forward transfer for the task $i$, denoted by $FT_{i}$, is formally defined as:
<p align="center">
$FT_{i}:=\frac{\text{AUC}_{i}-\text{AUC}_{i}^b}{1-\text{AUC}_{i}^b}$,
</p>
where:
<p align="center">
$\text{AUC}_{i}  :=\displaystyle\frac{1}{\Delta}\displaystyle \int_{(i-1)\cdot\Delta}^{i\cdot\Delta} p_{i}(t) \, dt$,
</p>
<p align="center">
$\text{AUC}_{i}^{b}  :=\displaystyle\frac{1}{\Delta}\displaystyle \int_{0}^{\Delta} p_{i}^{b}(t)\, dt$.
</p>
The average forward transfer for the entire sequence of tasks then defined as follows:
<p align="center">
$FT = \frac{1}{N} \sum_{i=1}^{N} FT_{i}$.
</p>

##### Forgetting

To quantify forgetting for task $i$, we measure the difference between the success rate on the task at the end of its training and the success rate on that task at the end of whole learning process, i.e.:
<p align="center">
$F_i=p_i(i\cdot\Delta)-p_i(T)$.
</p>
The average forgetting for the entire sequence of tasks is then defined as follows:
<p align="center">
$F=\frac{1}{N}\sum_{i=1}^{N}F_{i}$.
</p>

## Experiments

We perform experiments to compare PPO and SAC in terms of catastrophic forgetting and forward transfer, using a setup similar to the Continual World benchmark ([Wołczyk et al. 2021](https://arxiv.org/abs/2105.10919 "Wołczyk et al. 2021")). For this comparison, we create a sequence of $N=5$ tasks that the agent learns sequentially, without resetting the network parameters when transitioning between tasks. Although each task is evaluated throughout the learning process, it is only trained for $\Delta=10^6$ steps during its specific interval. We apply PPO and SAC with simple fine-tuning and three different CL methods: L2 regularization, elastic weight consolidation (EWC) ([Kirkpatrick et al. 2017](https://arxiv.org/abs/1612.00796 "Kirkpatrick et al. 2017")), and PackNet ([Mallya and Lazebnik 2018](https://arxiv.org/abs/1711.05769 "Mallya and Lazebnik 2018")).

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-0lax{text-align:left;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg"><thead>
  <tr>
    <th class="tg-baqh" rowspan="2">Method</th>
    <th class="tg-0lax" colspan="2">Performance</th>
    <th class="tg-0lax" colspan="2">Transfer</th>
    <th class="tg-0lax" colspan="2">Forgetting</th>
  </tr>
  <tr>
    <th class="tg-0pky">PPO</th>
    <th class="tg-0lax">SAC</th>
    <th class="tg-0lax">PPO</th>
    <th class="tg-0lax">SAC</th>
    <th class="tg-0lax">PPO</th>
    <th class="tg-0lax">SAC</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0lax">Fine-tuning</td>
    <td class="tg-0lax">-</td>
    <td class="tg-0lax">-</td>
    <td class="tg-0lax">-</td>
    <td class="tg-0lax">-</td>
    <td class="tg-0lax">-</td>
    <td class="tg-0lax">-</td>
  </tr>
</tbody>
</table>

While all CL methods address forgetting in PPO, it is observed that the average performance after training remains markedly inferior to that of SAC. Furthermore, it is noteworthy that PackNet with SAC demonstrates no signs of forgetting, whereas PackNet with PPO displays some degree of forgetting. It is postulated that this phenomenon occurs due to the utilisation of the replay buffer for retraining subsequent to the pruning phase, and the PPO training objective is not optimally aligned with the undertaking of multiple gradient steps on the same data. Finally, it was observed that fine-tuning with PPO did not exhibit any forward transfer, indicating that the knowledge gained from previous tasks was not beneficial when training on the current task. Conversely, higher forward transfer was observed in PPO than in SAC when CL methods were employed.
