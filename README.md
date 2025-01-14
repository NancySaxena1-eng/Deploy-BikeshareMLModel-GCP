# Deploy-BikeshareMLModel-GCP

Deploy the prediction serving application of this model to cloud run so that it can be used for online predictions by the business application(s)
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
To run the service in Cloud Run (GCP):
Follow the steps below:
1. Run model.ipynb and create the pickle file.
There is no pickle file loaded on github because the file was larger that 100MB.
2. Run these on the cmd:
docker build -t regression-model .

docker tag xgboost_coupon_model gcr.io/<project id>/regression-model

docker push gcr.io/<project id>/regression-model

gcloud run deploy regression-model --image  gcr.io/<project id/regression-model --region us-central1


Following permissions are needed before running the service on google cloud using cloud run:
# Assign Service account user role to the service account 
gcloud projects add-iam-policy-binding udemy-mlops \
--member=<service account number > --role=storage.buckets.get


# Assign Cloud Run role to the service account 
gcloud projects add-iam-policy-binding udemy-mlops \
  --member=<service account number> --role=roles/run.admin

# Command to run the build using cloudbuild.yaml
gcloud builds submit --region us-central1

*This project was a part of udemy course by Sid Raghunath
gcloud builds submit --region us-central1
