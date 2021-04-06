# Pill-Classification

## Summary
1. Tensorflow와 Keras를 활용하여 알약 인식 모델을 생성하고 학습한다. 결과물로 모델파일(model.json) 과 가중치 파일(weight.h5)을 생성한다.
2. 생성한 모델과 가중치를 사용해 AWS Lambda에서 실행하도록 function을 작성한다. 작동과정은 다음과 같다.
- AWS S3에 인식하고자 하는 이미지 파일을 업로드
- S3에 파일이 업로드 되었다는 Trigger 발생 시 작성한 Lambda function 수행
- Lambda 에서 미리 학습한 모델과 가중치를 통해 결과값 도출
- 결과값을 AWS DynamoDB에 저장

## Implementation
*Requirements*
- python3
- tensorflow
- numpy
- keras
- matplotlib
- boto3

