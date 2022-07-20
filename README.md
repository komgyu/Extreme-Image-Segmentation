# Extreme-Image-Segmentation
The goal of this project is to construct models for the extreme images and implement an algorithm to solve the decomposition problem. The problem construction includes two parts, the first is to use splines to simulate curves similar to tabular structures. The second is to use Voronoi tessellation to simulate cellular space separated only by thin membranes. In this project, seeded region growing (SRG) is applied to solve the decomposition of these two problem sets. The result is evaluated by Rand index and variation of information.

Generating splines

![image](https://user-images.githubusercontent.com/38833796/179972504-aec7fb27-896d-4798-a194-53e59ea20955.png)

Decomposition result

<img width="572" alt="bec1a4925ca98b1f8b2e9dceba19118" src="https://user-images.githubusercontent.com/38833796/179970923-40b8f94f-d47e-4622-92c2-060a374110bc.png">

Generating Voronoi cells

<img width="532" alt="5272ea967c01167b6d8cc2114ff6c72" src="https://user-images.githubusercontent.com/38833796/179971168-d6ef6e00-687e-4b87-9906-eda34fae3f28.png">

Decomposition result

<img width="807" alt="3ce58c9a733b9de4ae1e34127bf69b4" src="https://user-images.githubusercontent.com/38833796/179971352-83fa0be9-c39c-4246-87d3-d6ca88b5c16b.png">
