## Get_Virtual_Screen_from_frame 

#Small description
Make the screen someone holding visualize, you may track the real time screen content and display as a virtual screen synchronously.

#Implementation
Apply HOPENET to calculate the Euler angles in each frame of the videos, which is open-sourced on github, and use FIR filter to stabilize the Euler angles, resulting in a less shaky virtual screen.

#Demo Videos 🎥 
https://github.com/samuelzng/Get_Virtual_Screen_from_frame/blob/main/processed_test.mp4
<img width="381" height="266" alt="sample_case" src="https://github.com/user-attachments/assets/41ec08aa-d530-4361-8423-7ee2cfa33317" />
#Some concerns
1. 6D-repnet could be a superior alternative
   
2. Quaternions prevent Gimbol lock from Euler angles (Pending maintenance)
   

