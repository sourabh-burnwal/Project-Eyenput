# Project-Eyenput

### About:
  - This project is focussed on providing a software-based solution for Eye-tracking. Many differently-abled people are not able to use a Computer or play games on it. They also can't afford a costly hardware for the same. The Eyenput thus focuses particularly on a software solution.
  
### Workflow (yet):
  - Filtering out the rest and tracking the Eyeball
  - Blink detection and counting
  - Head pose estimation
  - Personalization of the software according to a particular user
  
### Repo Hierarchy:
    ```
    |
    |---Dumps (Trial-and-Error python scripts)
    |
    |---Finalized scripts
    |         |---Eyenput GUI.py (The Gui handler script or the front-end)
    |         |---eyenput_process.py (The back-end python code to handle the operations)
    |
    |---Images (Screenshot of the application and the Logo)
    |
    |---PyQt5 ui
    |     |---TheEyenput.ui (PyQt5 Designer output of the application)
    |
    |_
    
    ```
  
### Progress:
  - Appliation interface designing
  - Dlib's 68 landmarks detector has been used to filter out the eyeball
  - Blink detection and counting


### A sneek-peak of the Application under development:

![alt text](https://github.com/sourabh-burnwal/Project-Eyenput/blob/master/Images/The%20Eyenput%20application.png)
