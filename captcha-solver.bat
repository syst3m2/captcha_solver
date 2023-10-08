call docker run -d --rm --name "captcha-solver" -v "C:\Users\ntech\OneDrive\Documents\Coding\Captcha_Solver_App\captchaSolverModel":"/tmp/mounted_model/0001" -p 8502:8501 -t gcr.io/cloud-devrel-public-resources/gcloud-container-1.14.0:latest -d

call activate CY3650

python C:\Users\ntech\OneDrive\Documents\Coding\Captcha_Solver_App\npsCaptchaSolver.py

call conda deactivate

call docker stop captcha-solver