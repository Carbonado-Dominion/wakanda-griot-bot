import os
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import genai_core.utils.delete_files_with_prefix
from datetime import datetime

PROCESSING_BUCKET_NAME = os.environ["PROCESSING_BUCKET_NAME"]
UPLOAD_BUCKET_NAME = os.environ["UPLOAD_BUCKET_NAME"]
WORKSPACES_TABLE_NAME = os.environ["WORKSPACES_TABLE_NAME"]
DOCUMENTS_TABLE_NAME = os.environ.get("DOCUMENTS_TABLE_NAME")
DEFAULT_KENDRA_S3_DATA_SOURCE_BUCKET_NAME = os.environ.get(
    "DEFAULT_KENDRA_S3_DATA_SOURCE_BUCKET_NAME"
)

WORKSPACE_OBJECT_TYPE = "workspace"

dynamodb = boto3.resource("dynamodb")


def delete_workspace(workspace: dict):
    workspace_id = workspace["workspace_id"]
    genai_core.utils.delete_files_with_prefix.delete_files_with_prefix(
        UPLOAD_BUCKET_NAME, workspace_id
    )
    genai_core.utils.delete_files_with_prefix.delete_files_with_prefix(
        PROCESSING_BUCKET_NAME, workspace_id
    )
    genai_core.utils.delete_files_with_prefix.delete_files_with_prefix(
        DEFAULT_KENDRA_S3_DATA_SOURCE_BUCKET_NAME, f"documents/{workspace_id}"
    )
    genai_core.utils.delete_files_with_prefix.delete_files_with_prefix(
        DEFAULT_KENDRA_S3_DATA_SOURCE_BUCKET_NAME, f"metadata/documents/{workspace_id}"
    )

    workspaces_table = dynamodb.Table(WORKSPACES_TABLE_NAME)
    documents_table = dynamodb.Table(DOCUMENTS_TABLE_NAME)

    items_to_delete = []
    last_evaluated_key = None
    while True:
        query_args = {
            "KeyConditionExpression": boto3.dynamodb.conditions.Key("workspace_id").eq(
                workspace_id
            )
        }

        if last_evaluated_key:
            query_args["ExclusiveStartKey"] = last_evaluated_key

        response = documents_table.query(**query_args)
        items_to_delete.extend(response["Items"])

        last_evaluated_key = response.get("LastEvaluatedKey")
        if not last_evaluated_key:
            break

    # Batch delete in groups of 25
    for i in range(0, len(items_to_delete), 25):
        with documents_table.batch_writer() as batch:
            for item in items_to_delete[i : i + 25]:
                batch.delete_item(
                    Key={
                        "workspace_id": item["workspace_id"],
                        "document_id": item["document_id"],
                    }
                )

    print(f"Deleted {len(items_to_delete)} items.")

    response = workspaces_table.delete_item(
        Key={"workspace_id": workspace_id, "object_type": WORKSPACE_OBJECT_TYPE},
    )

    print(f"Delete Item succeeded: {response}")


def delete_kendra_document(workspace_id: str, document: dict):
    document_id = document["document_id"]
    document_vectors = document["vectors"]
    documents_diff = 1
    document_size_in_bytes = document["size_in_bytes"]
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    document_type = document["document_type"]

    if document["path"]:
        upload_bucket_key = workspace_id + "/" + document["path"]
        genai_core.utils.delete_files_with_object_key.delete_files_with_object_key(
            UPLOAD_BUCKET_NAME, upload_bucket_key
        )

    processing_bucket_key = workspace_id + "/" + document_id

    genai_core.utils.delete_files_with_prefix.delete_files_with_prefix(
        PROCESSING_BUCKET_NAME, processing_bucket_key
    )

    deleteKendraDocument(workspace_id, document_id, document_type)

    documents_table = dynamodb.Table(DOCUMENTS_TABLE_NAME)
    workspaces_table = dynamodb.Table(WORKSPACES_TABLE_NAME)

    try:
        response = documents_table.delete_item(
            Key={
                "workspace_id": workspace_id,
                "document_id": document_id,
            }
        )
        print(f"Delete document succeeded: {response}")

        updateResponse = workspaces_table.update_item(
            Key={"workspace_id": workspace_id, "object_type": WORKSPACE_OBJECT_TYPE},
            UpdateExpression="ADD size_in_bytes :incrementValue, "
            + "documents :documentsIncrementValue, "
            + "vectors :vectorsIncrementValue SET updated_at=:timestampValue",
            ExpressionAttributeValues={
                ":incrementValue": -document_size_in_bytes,
                ":documentsIncrementValue": -documents_diff,
                ":vectorsIncrementValue": -document_vectors,
                ":timestampValue": timestamp,
            },
            ReturnValues="UPDATED_NEW",
        )
        print(f"Workspaces table updated for the document: {updateResponse}")

    except (BotoCoreError, ClientError) as error:
        print(f"An error occurred: {error}")


def deleteKendraDocument(workspace_id, document_id, document_type):
    if document_type == "text":
        processing_object_key = f"{workspace_id}/{document_id}/content.txt"
        kendra_object_key = f"documents/{processing_object_key}"
        kendra_metadata_key = (
            f"metadata/documents/{processing_object_key}.metadata.json"
        )
        genai_core.utils.delete_files_with_prefix.delete_files_with_prefix(
            DEFAULT_KENDRA_S3_DATA_SOURCE_BUCKET_NAME, kendra_object_key
        )
        genai_core.utils.delete_files_with_prefix.delete_files_with_prefix(
            DEFAULT_KENDRA_S3_DATA_SOURCE_BUCKET_NAME, kendra_metadata_key
        )
