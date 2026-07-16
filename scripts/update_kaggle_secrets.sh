mkdir -p /tmp/kaggle-secrets
jq --arg id "$KAGGLE_USERNAME/secrets" '.id = $id' \
    secrets/dataset-metadata_template.json > /tmp/kaggle-secrets/dataset-metadata.json

jq --arg token "$HF_TOKEN" --arg user "$HF_USER" --arg spacename "$SPACE_NAME" \
   --arg repo "$MODEL_REPO" --arg metrics "$METRICS_REPO" \
   --arg synth "$SYNTH_DATA_REPO" --arg space "$SPACE_URL" \
   --arg company "$COMPANY" --arg desc "$COMPANY_DESC" \
   '.HF_TOKEN = $token | .HF_USER = $user | .MODEL_REPO = $repo | .METRICS_REPO = $metrics | .SYNTH_DATA_REPO = $synth | .SPACE_URL = $space | .SPACE_NAME = $spacename | .COMPANY = $company | .COMPANY_DESC = $desc' \
   secrets/secrets_template.json > /tmp/kaggle-secrets/secrets.json

if kaggle datasets metadata -d "$KAGGLE_USERNAME/secrets" &>/dev/null; then
    kaggle datasets version -p /tmp/kaggle-secrets -m "update"
else
    kaggle datasets create -p /tmp/kaggle-secrets
fi
