# Senior-Project
AI-generated Image Detection Tool
REQUIREMENTS: Windows OS, Python (current version)


Instructions to run software (on a WINDOWS based system only):

		Download and unzip the main branch files before beginning

1. Open your computer's command prompt

2. Go to the 'Senior-Project-main' folder and copy the directory

		Example: C:\Users\henri\Desktop\Senior-Project-main

3. On the command prompt type: 

		cd "paste-your-directory" (example: cd C:\Users\henri\Desktop\Senior-Project-main)

4. Set up a virtual environment by running the following command: 

		python -m venv venv

5. Start the virtual environment: 

		venv\Scripts\activate

6. Install the requirements and extensions: 

		pip install -r requirements.txt

7. Before running the application, edit line 9 on the file "image_classifier_app.py" which reads:

		model_path = r"C:\Users\Henrique\Desktop\Senior Project\efficientnet_finetuned.pth"

Change the model_path so that it correctly describes where the file named "efficientnet_finetuned.pth" is located. For example:

		model_path = r"C:\Users\Tommas\Documents\Senior-Project-main\efficientnet_finetuned.pth"

  This will be individual to each user, based on where the file is located on their computer.

8. Run the application:
		
  		python image_classifier_app.py

9. Click the upload button (logo), and select whichever image you wish to analyze
   

IMPORTANT NOTE: the application works only with common images formats such as .JPEG or .PNG
