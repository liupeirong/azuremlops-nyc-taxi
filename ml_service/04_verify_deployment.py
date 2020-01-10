import requests
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--uri',
        type=str,
        dest='uri',
        help='the scoring uri of the deployed service')
    parser.add_argument(
        '--token',
        type=str,
        dest='token',
        help='auth token to access the service')

    args = parser.parse_args()
    response = test_endpoint(args.uri, args.token)
    if response.status_code != 200:
        return 1
    else:
        return 0


def test_endpoint(uri, token):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + token
        }
    test_sample = json.dumps({'data': [
        [1,1,1.00,-73.957909,40.670761,-73.952194,40.662312,8.15,1,17,5,1]  # noqa: E231, E501
    ]})
    # test_sample = json.dumps({'data': score_df.values.tolist()})

    response = requests.post(uri, data=test_sample, headers=headers)
    print(response.status_code)
    print(response.json())
    return response


if __name__ == '__main__':
    ret = main()
    exit(ret)
