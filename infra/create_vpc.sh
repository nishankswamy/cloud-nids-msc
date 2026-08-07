#!/usr/bin/env bash
# Creates the network layer for the NIDS inference function.
# Deliberately excludes a NAT Gateway (~$32/month). S3 reachability is
# provided by a gateway endpoint, which is free.
set -euo pipefail

REGION=eu-central-1
NAME=nids

VPC_ID=$(aws ec2 create-vpc --cidr-block 10.0.0.0/16 \
  --tag-specifications "ResourceType=vpc,Tags=[{Key=Name,Value=${NAME}-vpc}]" \
  --query 'Vpc.VpcId' --output text --region $REGION)
echo "VPC: $VPC_ID"

SUBNET_A=$(aws ec2 create-subnet --vpc-id $VPC_ID \
  --cidr-block 10.0.1.0/24 --availability-zone ${REGION}a \
  --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=${NAME}-private-a}]" \
  --query 'Subnet.SubnetId' --output text --region $REGION)

SUBNET_B=$(aws ec2 create-subnet --vpc-id $VPC_ID \
  --cidr-block 10.0.2.0/24 --availability-zone ${REGION}b \
  --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=${NAME}-private-b}]" \
  --query 'Subnet.SubnetId' --output text --region $REGION)
echo "Subnets: $SUBNET_A $SUBNET_B"

RT_ID=$(aws ec2 create-route-table --vpc-id $VPC_ID \
  --tag-specifications "ResourceType=route-table,Tags=[{Key=Name,Value=${NAME}-private-rt}]" \
  --query 'RouteTable.RouteTableId' --output text --region $REGION)
aws ec2 associate-route-table --route-table-id $RT_ID --subnet-id $SUBNET_A --region $REGION >/dev/null
aws ec2 associate-route-table --route-table-id $RT_ID --subnet-id $SUBNET_B --region $REGION >/dev/null
echo "Route table: $RT_ID (no internet gateway, no NAT)"

ENDPOINT_ID=$(aws ec2 create-vpc-endpoint --vpc-id $VPC_ID \
  --service-name com.amazonaws.${REGION}.s3 \
  --vpc-endpoint-type Gateway --route-table-ids $RT_ID \
  --query 'VpcEndpoint.VpcEndpointId' --output text --region $REGION)
echo "S3 gateway endpoint: $ENDPOINT_ID"

SG_ID=$(aws ec2 create-security-group --group-name ${NAME}-lambda-sg \
  --description "NIDS Lambda: egress to S3 endpoint only" \
  --vpc-id $VPC_ID --query 'GroupId' --output text --region $REGION)
aws ec2 revoke-security-group-egress --group-id $SG_ID \
  --protocol -1 --port -1 --cidr 0.0.0.0/0 --region $REGION >/dev/null 2>&1 || true

PREFIX_LIST=$(aws ec2 describe-prefix-lists --region $REGION \
  --filters "Name=prefix-list-name,Values=com.amazonaws.${REGION}.s3" \
  --query 'PrefixLists[0].PrefixListId' --output text)
aws ec2 authorize-security-group-egress --group-id $SG_ID \
  --ip-permissions "IpProtocol=tcp,FromPort=443,ToPort=443,PrefixListIds=[{PrefixListId=$PREFIX_LIST}]" \
  --region $REGION >/dev/null
echo "Security group: $SG_ID (egress: HTTPS to S3 prefix list only)"

cat > infra/vpc_resources.json << JSON
{
  "vpc_id": "$VPC_ID",
  "subnet_a": "$SUBNET_A",
  "subnet_b": "$SUBNET_B",
  "route_table": "$RT_ID",
  "s3_endpoint": "$ENDPOINT_ID",
  "security_group": "$SG_ID",
  "region": "$REGION"
}
JSON
echo "Saved to infra/vpc_resources.json"
