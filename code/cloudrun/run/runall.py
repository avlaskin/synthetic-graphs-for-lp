import os


if __name__ == "__main__":
    for i in range(2,27):
        cmd = 'gcloud beta run jobs execute job-%d-n2v --region us-east1' % i
        #cmd = 'gcloud beta run jobs create job-%d-n2v --image gcr.io/gnn-test/n2v-job --tasks 10 --max-retries 2 --region us-east1 --cpu 8 --memory 32G --task-timeout=3000 --parallelism=2 --set-env-vars GRAPH_INDEX=%d' % (i, i)
        os.system(cmd)
