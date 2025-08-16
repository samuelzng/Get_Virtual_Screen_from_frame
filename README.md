<img width="762" height="533" alt="截屏2025-08-16 17 14 11" src="https://github.com/user-attachments/assets/788f4812-6241-435f-9e10-847c38ec49af" /><img width="762" height="533" alt="截屏2025-08-16 17 14 11" src="https://github.com/user-attachments/assets/41ec08aa-d530-4361-8423-7ee2cfa33317" />## Get_Virtual_Screen_from_frame 

#Small description
Make the screen someone holding visualize, you may track the real time screen content and display as a virtual screen synchronously.

#Implementation
Apply HOPENET to calculate the Euler angles in each frame of the videos, which is open-sourced on github, and use FIR filter to stabilize the Euler angles, resulting in a less shaky virtual screen.

#Demo Videos 🎥 
https://github.com/samuelzng/Get_Virtual_Screen_from_frame/blob/main/processed_test.mp4
![sample]()
#Some concerns
1. 6D-repnet could be a superior alternative
   
2. Quaternions prevent Gimbol lock from Euler angles (Pending maintenance)
   

