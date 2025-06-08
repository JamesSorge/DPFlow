DIR_KUBRIC="datasets/kubric_nk"
if [ ! -d "$DIR_KUBRIC" ]; then
    echo "Creating kubric_nk directory"
    mkdir -p ./datasets/kubric_nk
    echo "Done."
else
    echo "kubric_nk directory exists."
fi

if [ -z "$(ls -A datasets/kubric_nk)" ]; then
    # Download files for training/testing
    echo "Downloading Kubric-NK dataset files."
    # echo "Please download Kubric-NK dataset from the followwing Google Drive link: https://drive.google.com/drive/folders/1vSShkqyYwLJYX38iJP3kg6x-DMpCn0R6"

    # The following option failed; it appears to work, and downloads an empty zip file.
    # # configs_1k.zip
    # wget --no-check-certificate "https://drive.usercontent.google.com/download?id=136Zl6mTn9CUUMoIS6lBKmVdVmCQg3dFw&export=download&confirm=t&uuid=cd3f32ea-a48d-4d36-860f-7563d5b59e01" -O configs_1k.zip
    # # configs_2k.zip
    # wget --no-check-certificate "https://drive.usercontent.google.com/download?id=1pkRO3IwjUGwdx6S0ffa8yaX81DKL-lhx&export=download&confirm=t&uuid=cc3d0699-3085-4838-b7d2-545ab7a93a45"
    # # configs_4k.zip
    # wget --no-check-certificate "https://drive.usercontent.google.com/download?id=1cPPRKHgMY8A1A2KQzdFibxtrkwQmE815&export=download&confirm=t&uuid=8048c3d8-544c-4c3d-aef3-6269fb5507d2"
    # # configs_8k.zip
    # wget --no-check-certificate "https://drive.usercontent.google.com/download?id=1wno4bNpWReeopNPgrmHeXJwBaGIiJmtz&export=download&confirm=t&uuid=fe7a6eba-90ff-4020-808c-8fae374d6a59"

    # # rgba_1k.zip
    # wget --no-check-certificate "https://drive.usercontent.google.com/download?id=16Wz7FQoP3r_l-vAfMgSWPdePG99MqQ_C&export=download&confirm=t&uuid=13297a3b-1894-4b88-b7c4-45ed9ed2e068" -O rgba_1k.zip
    # # rgba_2k.zip
    # wget --no-check-certificate "https://drive.usercontent.google.com/download?id=1_vXM6AyAswqfyxor5OvbROxx4g-LOpY4&export=download&confirm=t&uuid=74900d06-a029-4d91-a472-51f0040a08bd"
    # # rgba_4k.zip
    # wget --no-check-certificate "https://drive.usercontent.google.com/download?id=10wWQ_eT0ObKlf_nrxkYy3deGpMVlzWOC&export=download&confirm=t&uuid=1d09c051-90fb-49fa-a48a-7daeaf4899b7"
    # # rgba_8k.zip
    # wget --no-check-certificate "https://drive.usercontent.google.com/download?id=1gfOrvOhMAgXoHg1ciyY3dKuO6dbn5imM&export=download&confirm=t&uuid=05777a76-f856-456e-902b-cee3faff0fda"

    # # forward_flow_1k.zip
    # wget --no-check-certificate "https://drive.usercontent.google.com/download?id=1x9o8XP4sfdu9XezaVU5bTJW75XJwMj8P&export=download&confirm=t&uuid=840516e0-5b3a-46fe-9635-15fd1dd86422" -O forward_flow_1k.zip
    # # forward_flow_2k.zip
    # wget --no-check-certificate "https://drive.usercontent.google.com/download?id=146yJ-NfZZ_p7Off631heKvVh8dckDXZV&export=download&confirm=t&uuid=19ae331d-9522-4a88-9cd7-cf30ff5baaec"
    # # forward_flow_4k.zip
    # wget --no-check-certificate "https://drive.usercontent.google.com/download?id=1XPoF1xW94Faaz9xvfBJTdOkhv9M9dUIy&export=download&confirm=t&uuid=62fb4e86-e313-4b51-9c82-a981ffe1d9c9"
    # # forward_flow_8k.zip
    # wget --no-check-certificate ""

    # Backward flow 1k
    # wget --no-check-certificate "https://drive.usercontent.google.com/download?id=15LOSIQSO6AvQXTT-K-U_47eHZEnr8sWJ&export=download&authuser=0&confirm=t&uuid=fe948903-e8d7-464a-8895-f99d8e4bcbfc&at=ALoNOgmJh_SYbyOsIc71VOjXJecO:1749344728717" -O backward_flow_1k.zip

    # Move all downloaded files to the correct directory
    # mv rgba*.zip datasets/kubric_nk
    # mv config*.zip datasets/kubric_nk
    # mv forward_flow*.zip datasets/kubric_nk
    # mv backward_flow*.zip datasets/kubric_nk
else
    wget --no-check-certificate "https://drive.usercontent.google.com/download?id=15LOSIQSO6AvQXTT-K-U_47eHZEnr8sWJ&export=download&authuser=0&confirm=t&uuid=fe948903-e8d7-464a-8895-f99d8e4bcbfc&at=ALoNOgmJh_SYbyOsIc71VOjXJecO:1749344728717" -O backward_flow_1k.zip
    mv backward_flow*.zip datasets/kubric_nk
    echo "Dataset files present."
fi