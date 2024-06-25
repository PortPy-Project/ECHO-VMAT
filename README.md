<h1 align="center">
  <img src="./images/ECHO-VMAT.png" width=70% height="40%">
</h1>

# ECHO-VMAT
ECHO, which stands for **E**xpedited **C**onstrained **H**ierarchical **O**ptimization, is an internal automated treatment planning tool developed by our team. As of now, ECHO for Intensity-Modulated Radiation Therapy (IMRT) has been clinically employed to treat over 10,000 patients and remains a critical tool in our daily clinical operations. We have recently extended ECHO's capabilities to encompass Volumetric Modulated Arc Therapy (VMAT), and this extension is currently being utilized in clinical practice.

We are pleased to announce Sequential Convex Programming (SCP) based VMAT optimization on the main PortPy repository [vmat_scp_tutorial.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/vmat_scp_tutorial.ipynb). Users are encouraged to use [PortPy](https://github.com/PortPy-Project/PortPy) package to conduct their research on VMAT planning. 

ECHO-VMAT optimization is an extension of the SCP-based VMAT optimization, incorporating hierarchical constrained optimization to address conflicting objectives between the target and OAR.
By the end of 2024, we aim to share the source code for ECHO-VMAT through the PortPy platform. This release will be based on the following two research papers that provide extensive details on the underlying methodology. You can find the slides of our AAPM-2023 talk [here](https://github.com/PortPy-Project/ECHO-VMAT/blob/main/VMAT-AAPM-2023.pptx).

1- *Automated VMAT treatment planning using sequential convex programming: algorithm development and clinical implementation*, 
Pınar Dursun, Linda Hong, Gourav Jhanwar, Qijie Huang, Ying Zhou, Jie Yang, Hai Pham, Laura Cervino, Jean M Moran, Joseph O Deasy, Masoud Zarepisheh,
**Physics in Medicine & Biology**, [link](https://iopscience.iop.org/article/10.1088/1361-6560/ace09e/pdf)

2- *Solving the volumetric modulated arc therapy (VMAT) problem using a sequential convex programming method*, 
Pınar Dursun, Masoud Zarepisheh, Gourav Jhanwar and Joseph O Deasy
**Physics in Medicine and Biology**, [link](https://iopscience.iop.org/article/10.1088/1361-6560/abee58/pdf)
