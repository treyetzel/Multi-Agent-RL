# Multi-Agent Reinforcement Learning

This repository is for experimenting with multi-agent reinforcement algorithms using pytorch and pettingzoo.

<!--
[![](https://img.shields.io/badge/-Training%20Results-informational?style=for-the-badge)](https://wandb.ai/koulanurag/minimal-marl/reports/Minimal-Marl--Vmlldzo4MzM2MDc?accessToken=vy6dydemfdvekct02pevp3girjvb0tnt1ou2acb2h0fl478hdjqqu8ydbco6uz38)
[![](https://img.shields.io/badge/-Work%20in%20Progress-orange?style=for-the-badge)]()
-->
## Installation

install pytorch and cuda here: https://pytorch.org/get-started/locally/
 
  
 then install the following
```bash 
pip install supersuit
pip install pettingzoo[all]
```
wandb is optional, you can create an account [here](https://wandb.ai/site), or set --wandb to False when running the training file
```bash 
pip install wandb
```
## Usage

```bash
python train.py --wandb False
