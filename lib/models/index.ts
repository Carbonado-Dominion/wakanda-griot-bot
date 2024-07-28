import * as cdk from "aws-cdk-lib";
import * as ssm from "aws-cdk-lib/aws-ssm";
import { Construct } from "constructs";
import * as iam from "aws-cdk-lib/aws-iam";
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import { Shared } from "../shared";
import {
  Modality,
  ModelInterface,
  SageMakerModelEndpoint,
  SupportedSageMakerModels,
  SystemConfig,
} from "../shared/types";
import {
  HuggingFaceSageMakerEndpoint,
  SageMakerInstanceType,
  DeepLearningContainerImage,
  JumpStartSageMakerEndpoint,
  JumpStartModel,
} from "@cdklabs/generative-ai-cdk-constructs";
import { NagSuppressions } from "cdk-nag";
import { createStartSchedule, createStopSchedule } from "./sagemaker-schedule"
import {ContainerImages} from "../sagemaker-model/container-images"

export interface ModelsProps {
  readonly config: SystemConfig;
  readonly shared: Shared;
}

export class Models extends Construct {
  public readonly models: SageMakerModelEndpoint[];
  public readonly modelsParameter: ssm.StringParameter;

  constructor(scope: Construct, id: string, props: ModelsProps) {
    super(scope, id);

    const models: SageMakerModelEndpoint[] = [];

    let hfTokenSecret: secretsmanager.Secret | undefined;
    if (props.config.llms.huggingfaceApiSecretArn) {
      hfTokenSecret = secretsmanager.Secret.fromSecretCompleteArn(this, 'HFTokenSecret', props.config.llms.huggingfaceApiSecretArn) as secretsmanager.Secret;
    }
    
    if (
      props.config.llms?.sagemaker.includes(SupportedSageMakerModels.FalconLite2)
    ) {
      const FALCON_LITE_2_MODEL_ID = "Amazon/FalconLite2";
      const FALCON_LITE_2_ENDPOINT_NAME = FALCON_LITE_2_MODEL_ID.split("/")
        .join("-")
        .split(".")
        .join("-");

      const falcon_lite_2 = new JumpStartSageMakerEndpoint(this, "Falcon-Lite2", {
        model: JumpStartModel.HUGGINGFACE_LLM_AMAZON_FALCONLITE2_1_1_0,
        instanceType: SageMakerInstanceType.ML_G5_48XLARGE,
        instanceCount: 3,
        endpointName: FALCON_LITE_2_ENDPOINT_NAME,
        environment: {
          MAX_INPUT_TOKENS: JSON.stringify(10000),
          MAX_TOTAL_TOKENS: JSON.stringify(24000),
          MAX_BATCH_TOTAL_TOKENS: JSON.stringify(24000),
          SM_NUM_GPUS: JSON.stringify(4),
          MAX_CONCURRENT_REQUESTS: JSON.stringify(4),
          SAGEMAKER_STREAMING_ENABLED: "true", // Enable streaming
          SAGEMAKER_CONTAINER_LOG_LEVEL: "INFO", // Optional: for better logging
        },
      });

      this.suppressCdkNagWarningForEndpointRole(falcon_lite_2.role);

      models.push({
        name: `jumpstart-${FALCON_LITE_2_ENDPOINT_NAME!}`,
        endpoint: falcon_lite_2.cfnEndpoint,
        responseStreamingSupported: true,
        inputModalities: [Modality.Text],
        outputModalities: [Modality.Text],
        interface: ModelInterface.LangChain,
        ragSupported: true,
      });
    }
    
    
    if (
      props.config.llms?.sagemaker.includes(SupportedSageMakerModels.Zephyr7b)
    ) {
      const ZEPHYR_7B_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta";
      const ZEPHYR_7B_ENDPOINT_NAME = ZEPHYR_7B_MODEL_ID.split("/")
        .join("-")
        .split(".")
        .join("-");

      const zephyr7b = new JumpStartSageMakerEndpoint(this, "Zephyr7b", {
        model: JumpStartModel.HUGGINGFACE_LLM_HUGGINGFACEH4_ZEPHYR_7B_BETA_1_1_0,
        instanceType: SageMakerInstanceType.ML_G5_24XLARGE,
        endpointName: ZEPHYR_7B_ENDPOINT_NAME,
        instanceCount: 3,
      });

      this.suppressCdkNagWarningForEndpointRole(zephyr7b.role);

      models.push({
        name: `jumpstart-${ZEPHYR_7B_ENDPOINT_NAME!}`,
        endpoint: zephyr7b.cfnEndpoint,
        responseStreamingSupported: false,
        inputModalities: [Modality.Text],
        outputModalities: [Modality.Text],
        interface: ModelInterface.LangChain,
        ragSupported: true,
      });
    }

    const modelsParameter = new ssm.StringParameter(this, "ModelsParameter", {
      stringValue: JSON.stringify(
        models.map((model) => ({
          name: model.name,
          endpoint: model.endpoint.endpointName,
          responseStreamingSupported: model.responseStreamingSupported,
          inputModalities: model.inputModalities,
          outputModalities: model.outputModalities,
          interface: model.interface,
          ragSupported: model.ragSupported,
        }))
      ),
    });

    this.models = models;
    this.modelsParameter = modelsParameter;
    
    if (models.length > 0 && props.config.llms?.sagemakerSchedule?.enabled) {

      let schedulerRole: iam.Role = new iam.Role(this, 'SchedulerRole', {
        assumedBy: new iam.ServicePrincipal('scheduler.amazonaws.com'),
        description: 'Role for Scheduler to interact with SageMaker',
      });
      
      schedulerRole.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess'));
      this.suppressCdkNagWarningForEndpointRole(schedulerRole);

      models.forEach((model) => {
          createStartSchedule(this, id, model.endpoint, schedulerRole, props.config);
          createStopSchedule(this, id, model.endpoint, schedulerRole, props.config);
      });
    }
  }

  private suppressCdkNagWarningForEndpointRole(role: iam.Role) {
    NagSuppressions.addResourceSuppressions(
      role,
      [
        {
          id: "AwsSolutions-IAM4",
          reason:
            "Gives user ability to deploy and delete endpoints from the UI.",
        },
        {
          id: "AwsSolutions-IAM5",
          reason:
            "Gives user ability to deploy and delete endpoints from the UI.",
        },
      ],
      true
    );
  }
}
