# Analysis and Evaluation of Grad-CAM Explanations
Final Project for DD2412 Course (Deep Learning, Advanced), KTH

The scope of this project was to reproduce the findings of [Grad-CAM](https://arxiv.org/abs/1610.02391), a deep visualization technique suitable to any CNN. We performed the following tasks:

1. Evaluated Grad-CAM on the Weakly Supervised Localisation Task (ILSVRC15 validation set), in which the agent aims to localise the object (via a bounding box) without being explicitly trained to do so, based solely on the visualization.
1. Compute Pointing Game Accuracy and Recall (ILSVRC15 validation set).
1. Compare Grad-CAM, Guided Grad-CAM and Guided Backpropagation with [occlusion maps](https://arxiv.org/pdf/1511.06457.pdf).
1. Reproduce and analyze a User study to compare the thetrustworthiness of Guided Grad-CAM and Guided Backpropagation using VGG-16 and AlexNet, leveraging the fact that the former is known to be more accurate.
1. Compare Grad-CAM with [Grad-CAM++](https://arxiv.org/abs/1710.11063), Integrated Gradients and SHAP medical data.
1. Propose a novel experiment for evaluating Grad-CAM's [sensitivity](https://arxiv.org/abs/1703.01365).
1. Compare Grad-CAM with Integrated Gradients and SHAP in regards to [contrastivity and fidelity](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pope_Explainability_Methods_for_Graph_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)

## Task 1



### Implemented method visualization
<img src="/images_for_readme/cat_dog.png" alt="Cat and Dog" title="Implemented Models">

In the first task, we managed to successfully reproduce the results of the original paper. We see that Grad-CAM manages noteworthy results in a task in which it was not excplicitly trained for.
### Task 1 results (original paper results in parentheses)
<img src="/images_for_readme/task 1 results.png" title="Task 1">

## Task 2
In the pointing game we examine the maximally activated point produced by the heatman and check whether it lies inside the real label's bounding box (accuracy). We also consider the recall, by allowing the model to renounce any top-5 visualization with a max activation below a given threshold.
### Task 2 results
<img src="/images_for_readme/task 2 results.png">


## Task 3

Measuring Rank Correlation between Occlusion maps and Grad-CAM, Guided Grad-CAM and guided Backpropagation. Relative to occlusion maps, Guided Grad-CAM is slightly more similar than Grad-CAM which is significantly more similar than guided backpropagation.
### Task 3 results
<img src="/images_for_readme/task 3 results.png">

<img src="/images_for_readme/occlusion.png">

## Task 4
In this user study, users were tasksd with choosing between the two agents and grading them on a scale from -2 to 2 (-2: A is substantially better ... 2: B is substantially better). Our results indicate that the user study conducted in the original paper is not robust enough, as seen by the high variance. 
### Task 4 results
<img src="/images_for_readme/task 4 results.png">

### Guided Grad-CAM
<img src="/images_for_readme/user_study_guided_cam.png">

### Guided Backpropagation
<img src="/images_for_readme/user_study_guided_bp.png">

## Task 5
In this task, we examined the efficacy of Grad-CAM and Grad-CAM++ with Integrated gradients and SHAP, using a DenseNet121 architecture pretrained on Chest-X-Ray14. We then measured the ratio of activated pixels (beyond 85 %) which lay within the target bounding boxes.
### Task 5 results
<img src="/images_for_readme/task 5 results.png">

<img src="/images_for_readme/xray_cam.png">
<img src="/images_for_readme/xray_camplus.png">
<img src="/images_for_readme/xray_ig.png">
<img src="/images_for_readme/xray_shap.png">

## Task 6
A visualization method is sensitive if it assigns non-zero significance to all features which are capable of singlehandedly change the prediction of the classifier. For this task, we generated single pixel attacks and analyzed Grad-CAM with VGG-16 and GoogLeNet. Empirical results indicate that Grad-CAM with GoogLeNet exhibits sensitivity.
### Task 6 results
<img src="/images_for_readme/task 6 results.png">

### Example of single pixel attack (baseball -> assault rifle)
<img src="/images_for_readme/single_pixel_attack.png">

## Task 7
We measure fidelity (highlighted feature relevance to result) and contrastivity (overlap between different class visualizations). Grad-CAM showcases the highest contrastivity.
### Task 7 results
<img src="/images_for_readme/task 7 results.png">

<img src="/images_for_readme/contrastivity_and_fidelity.png">

## Robustness to adversarial attacks
<img src="/images_for_readme/adversarial.png">
