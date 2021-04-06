import boto3
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import model_from_json
import numpy as np
import os

# Lambda 함수 별 환경변수에서 Key 값 받아옴
ACCESS_KEY = os.environ.get('ACCESS_KEY')
SECRET_KEY = os.environ.get('SECRET_KEY')

def downloadFromS3(strBucket, s3_path, local_path):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )
    s3_client.download_file(strBucket, s3_path, local_path)


def uploadToS3(bucket, s3_path, local_path):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )
    s3_client.upload_file(local_path, bucket, s3_path)


def predict(img_local_path):
    json_file = open('/---/model.json', "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('/---/model_weights.h5')
    img = image.load_img(img_local_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = loaded_model.predict(x)
    res = decode_predictions(preds)
    return res


def handler(event, context):
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    file_path = event['Records'][0]['s3']['object']['key']
    file_name = file_path.split('/')[-1]
    downloadFromS3(bucket_name, file_path, '/tmp/'+file_name)
    # 모델 json 파일 받아오기
    downloadFromS3(
        's3 버킷 이름',
        '다운로드 받은 파일의 절대 경로',
        '저장 위치'
    )
    # 가중치 담은 .h5파일을 s3에서 받아오기, 그냥 git에서 받아오면 비용 많이 발생.
    downloadFromS3(
        's3 버킷 이름',
        '다운로드 받은 파일의 절대 경로',
        '저장 위치'
    )
    result = predict('/tmp/'+file_name)

    #결과값을 다이나모DB에 올려서 확인하기
    _tmp_dic = {x[1]:{'N':str(x[2])} for x in result[0]}
    dic_for_dynamodb = {'M': _tmp_dic}
    dynamo_client = boto3.client(
        'dynamodb',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name='ap-northeast-2'  # DynamoDB는 리전 이름이 필요 'ap-northeast-2'는 아시아-태평양 서율 리전
    )
    dynamo_client.put_item(
        TableName='DynamoDB의 Table이름',
        Item={
            '해당 DB의 key value에 맞게 작성',
            dic_for_dynamodb
        }
    )
    return result