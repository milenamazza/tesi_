#!/bin/bash

CONFIG_FILE="pyproject.toml"
CSV_FILE="ris-7-adam.csv"

# Function to update a value in the TOML file
update_value() {
    local key="$1"
    local value="$2"
    local pattern="$key = .*"
    local replacement="$key = $value"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s|$pattern|$replacement|" "$CONFIG_FILE"
    else
        sed -i "s|$pattern|$replacement|" "$CONFIG_FILE"
    fi
}

# Function to add a parameter to the TOML file
add_parameter() {
    local key="$1"
    local value="$2"
    
    # Check if parameter already exists
    if grep -q "^$key = " "$CONFIG_FILE"; then
        update_value "$key" "$value"
    else
        # Add parameter after the strategy line
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "/^strategy = /a\\
$key = $value
" "$CONFIG_FILE"
        else
            sed -i "/^strategy = /a $key = $value" "$CONFIG_FILE"
        fi
    fi
}

# Function to remove a parameter from the TOML file
remove_parameter() {
    local key="$1"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "/^$key = /d" "$CONFIG_FILE"
    else
        sed -i "/^$key = /d" "$CONFIG_FILE"
    fi
}

# Function to clean all strategy-specific parameters
clean_strategy_params() {
    # Remove all strategy-specific parameters
    remove_parameter "proximal_mu"
    remove_parameter "eta"
    remove_parameter "beta_1"
    remove_parameter "beta_2"
    remove_parameter "reconstruction_weight"
    remove_parameter "classification_weight"
    remove_parameter "kld_weight"
    remove_parameter "beta_vae"
}

# Function to check if a combination has already been tested
is_combination_done() {
    local supernodes="$1"
    local latent_dim="$2"
    local hidden_dim="$3"
    local rounds="$4"
    local epochs="$5"
    local batch="$6"
    local lr="$7"
    local gamma="$8"
    local step="$9"
    local fraction="${10}"
    local strategy="${11}"
    shift 11  # Remove the first 11 arguments
    local strategy_params=("$@")  # Remaining arguments are strategy-specific parameters

    if [[ ! -f "$CSV_FILE" ]]; then
        return 1  # File doesn't exist, combination not done
    fi

    # Build the search pattern - we'll search for the key parameters that define a unique experiment
    # Format: Round,Clients,Latent Dim,Hidden Dim,Rounds,Epochs,Batch Size,Learning Rate,Gamma,Step,fraction-fit,Strategy,Recon Weight,Class Weight,KLD Weight,Beta VAE,Proximal Mu,Eta Adagrad,Eta Adam,Beta 1,Beta 2,Patience,Min Delta,Accuracy,F1 Score,Precision,Recall,Train Loss
    
    local search_pattern
    case $strategy in
        "fedvae")
            local recon_weight="${strategy_params[0]}"
            local class_weight="${strategy_params[1]}" 
            local kld_weight="${strategy_params[2]}"
            local beta_vae="${strategy_params[3]}"
            # Search for lines containing all the key parameters (not necessarily at the beginning)
            search_pattern=",$supernodes,$latent_dim,$hidden_dim,$rounds,$epochs,$batch,$lr,$gamma,$step,$fraction,$strategy,$recon_weight,$class_weight,$kld_weight,$beta_vae,0.0,0.0,0.0,0.0,0.0,"
            ;;
        "fedprox")
            local proximal_mu="${strategy_params[0]}"
            search_pattern=",$supernodes,$latent_dim,$hidden_dim,$rounds,$epochs,$batch,$lr,$gamma,$step,$fraction,$strategy,0.0,0.0,0.0,0.0,$proximal_mu,0.0,0.0,0.0,0.0,"
            ;;
        "fedadagrad")
            local eta="${strategy_params[0]}"
            search_pattern=",$supernodes,$latent_dim,$hidden_dim,$rounds,$epochs,$batch,$lr,$gamma,$step,$fraction,$strategy,0.0,0.0,0.0,0.0,0.0,$eta,0.0,0.0,0.0,"
            ;;
        "fedadam")
            local eta="${strategy_params[0]}"
            local beta_1="${strategy_params[1]}"
            local beta_2="${strategy_params[2]}"
            search_pattern=",$supernodes,$latent_dim,$hidden_dim,$rounds,$epochs,$batch,$lr,$gamma,$step,$fraction,$strategy,0.0,0.0,0.0,0.0,0.0,0.0,$eta,$beta_1,$beta_2,"
            ;;
        *)
            # For other strategies: all strategy params are 0.0
            search_pattern=",$supernodes,$latent_dim,$hidden_dim,$rounds,$epochs,$batch,$lr,$gamma,$step,$fraction,$strategy,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,"
            ;;
    esac
    
    # Use grep to search for the pattern (allowing any round number at the beginning)
    if grep -q "$search_pattern" "$CSV_FILE"; then
        echo "✓ Found existing combination: $search_pattern"
        return 0  # Found
    else
        echo "✗ New combination: $search_pattern"
        return 1  # Not found
    fi
}

# Parameters to test
NUM_SUPERNODES=("7")
NUM_SERVER_ROUNDS=("100")
LOCAL_EPOCHS=("5" "10" "15")
HIDDEN_DIMS=("512" "256")
LATENT_DIMS=("15" "10" "7")
LEARNING_RATES=("0.01" "0.001" "0.0001")
BATCH_SIZES=(  "32")  # Make sure this matches what you want to test
GAMMAS=("0.1")
STEPS=("10" "5")
FRACTIONS=("1.0")

# Available strategies
STRATEGIES=("fedadam")

# FedVAE specific parameters
RECONSTRUCTION_WEIGHTS=("0.2")
CLASSIFICATION_WEIGHTS=("0.2")
KLD_WEIGHTS=("1.0")
BETA_VAE_VALUES=("1.0")

# FedProx specific parameters
PROXIMAL_MUS=("0.01" "0.1" "1.0")

# FedAdagrad specific parameters
ETA_ADAGRAD_VALUES=("0.1" "0.5")

# FedAdam specific parameters
ETA_ADAM_VALUES=("0.001" "0.01")
BETA_1_VALUES=("0.9")
BETA_2_VALUES=("0.99")

# Early stopping parameters
PATIENCE_VALUES=("15")
MIN_DELTA_VALUES=("0.000")

# Calculate total combinations
# Calculate total combinations
calculate_combinations() {
    local total=0
    
    for strategy in "${STRATEGIES[@]}"; do
        case $strategy in
            "fedvae")
                # Calculate valid combinations (where recon_weight + class_weight <= 1.0)
                local valid_combinations=0
                for recon in "${RECONSTRUCTION_WEIGHTS[@]}"; do
                    for class in "${CLASSIFICATION_WEIGHTS[@]}"; do
                        # Check if weights sum <= 1.0 using awk instead of bc
                        if awk "BEGIN { exit !($recon + $class <= 1.0) }"; then
                            valid_combinations=$((valid_combinations + ${#KLD_WEIGHTS[@]} * ${#BETA_VAE_VALUES[@]}))
                        fi
                    done
                done
                total=$((total + valid_combinations))
                ;;
            "fedprox")
                total=$((total + ${#PROXIMAL_MUS[@]}))
                ;;
            "fedadagrad")
                total=$((total + ${#ETA_ADAGRAD_VALUES[@]}))
                ;;
            "fedadam")
                local adam_combinations=$(( ${#ETA_ADAM_VALUES[@]} * ${#BETA_1_VALUES[@]} * ${#BETA_2_VALUES[@]} ))
                total=$((total + adam_combinations))
                ;;
            *)
                total=$((total + 1))
                ;;
        esac
    done
    
    total=$((total * ${#NUM_SUPERNODES[@]} * ${#NUM_SERVER_ROUNDS[@]} * ${#LOCAL_EPOCHS[@]} * ${#HIDDEN_DIMS[@]} * ${#LATENT_DIMS[@]} * ${#LEARNING_RATES[@]} * ${#BATCH_SIZES[@]} * ${#GAMMAS[@]} * ${#STEPS[@]} * ${#FRACTIONS[@]} * ${#PATIENCE_VALUES[@]} * ${#MIN_DELTA_VALUES[@]}))
    
    echo $total
}

TOTAL_COMBINATIONS=$(calculate_combinations)
COUNT=0

echo "🚀 Starting experiments with $TOTAL_COMBINATIONS total combinations"
echo "📊 Current CSV file has $(wc -l < "$CSV_FILE" 2>/dev/null || echo 0) lines"

# Main loop
for supernodes in "${NUM_SUPERNODES[@]}"; do
    for rounds in "${NUM_SERVER_ROUNDS[@]}"; do
        for epochs in "${LOCAL_EPOCHS[@]}"; do
            for hidden_dim in "${HIDDEN_DIMS[@]}"; do
                for latent_dim in "${LATENT_DIMS[@]}"; do
                    for lr in "${LEARNING_RATES[@]}"; do
                        for batch in "${BATCH_SIZES[@]}"; do
                            for gamma in "${GAMMAS[@]}"; do
                                for step in "${STEPS[@]}"; do
                                    for fraction in "${FRACTIONS[@]}"; do
                                        for patience in "${PATIENCE_VALUES[@]}"; do
                                            for min_delta in "${MIN_DELTA_VALUES[@]}"; do
                                                for strategy in "${STRATEGIES[@]}"; do
                                                    
                                                    case $strategy in
                                                        "fedvae")
                                                            for recon_weight in "${RECONSTRUCTION_WEIGHTS[@]}"; do
                                                                for class_weight in "${CLASSIFICATION_WEIGHTS[@]}"; do
                                                                    # Skip if weights sum > 1.0 - using awk instead of bc
                                                                    if awk "BEGIN { exit !($recon_weight + $class_weight > 1.0) }"; then
                                                                        echo "⚠️  Skipping invalid FedVAE combination: recon_weight($recon_weight) + class_weight($class_weight) = $(awk "BEGIN { print $recon_weight + $class_weight }") > 1.0"
                                                                        continue
                                                                    fi
                                                                    
                                                                    for kld_weight in "${KLD_WEIGHTS[@]}"; do
                                                                        for beta_vae in "${BETA_VAE_VALUES[@]}"; do
                                                                            ((COUNT++))
                                                                            echo ""
                                                                            echo "🔁 Combination $COUNT / $TOTAL_COMBINATIONS - $strategy"
                                                                            echo "   Parameters: clients=$supernodes, latent=$latent_dim, hidden=$hidden_dim, rounds=$rounds, epochs=$epochs, batch=$batch"
                                                                            echo "   FedVAE: recon=$recon_weight, class=$class_weight, kld=$kld_weight, beta=$beta_vae"
                                                                            
                                                                            # Check if this specific combination is done
                                                                            if is_combination_done "$supernodes" "$latent_dim" "$hidden_dim" "$rounds" "$epochs" "$batch" "$lr" "$gamma" "$step" "$fraction" "$strategy" "$recon_weight" "$class_weight" "$kld_weight" "$beta_vae"; then
                                                                                echo "⏩  Skipping already completed combination"
                                                                                continue
                                                                            fi
                                                                            
                                                                            # Clean all strategy params first
                                                                            clean_strategy_params
                                                                            
                                                                            # Update base parameters
                                                                            update_value "options.num-supernodes" "$supernodes"
                                                                            update_value "num-server-rounds" "$rounds"
                                                                            update_value "local-epochs" "$epochs"
                                                                            update_value "num-supernodes" "$supernodes"
                                                                            update_value "hidden-dim" "$hidden_dim"
                                                                            update_value "latent-dim" "$latent_dim"
                                                                            update_value "learning-rate" "$lr"
                                                                            update_value "batch-size" "$batch"
                                                                            update_value "gamma" "$gamma"
                                                                            update_value "step" "$step"
                                                                            update_value "fraction-fit" "$fraction"
                                                                            update_value "strategy" "\"$strategy\""
                                                                            update_value "patience" "$patience"
                                                                            update_value "min_delta" "$min_delta"
                                                                            
                                                                            # Add FedVAE specific parameters
                                                                            add_parameter "reconstruction_weight" "$recon_weight"
                                                                            add_parameter "classification_weight" "$class_weight"
                                                                            add_parameter "kld_weight" "$kld_weight"
                                                                            add_parameter "beta_vae" "$beta_vae"
                                                                            
                                                                            echo "▶️ RUNNING EXPERIMENT..."
                                                                            
                                                                            timeout 7600 flwr run
                                                                            exit_code=$?
                                                                            if [ $exit_code -eq 124 ]; then
                                                                                echo "⚠️  Timeout for this combination"
                                                                            elif [ $exit_code -ne 0 ]; then
                                                                                echo "❌  Error occurred (exit code: $exit_code)"
                                                                            else
                                                                                echo "✅  Experiment completed successfully"
                                                                            fi
                                                                        done
                                                                    done
                                                                done
                                                            done
                                                            ;;
                                                            
                                                        "fedprox")
                                                            for proximal_mu in "${PROXIMAL_MUS[@]}"; do
                                                                ((COUNT++))
                                                                echo "🔁 Combination $COUNT / $TOTAL_COMBINATIONS - $strategy"
                                                                
                                                                if is_combination_done "$supernodes" "$latent_dim" "$hidden_dim" "$rounds" "$epochs" "$batch" "$lr" "$gamma" "$step" "$fraction" "$strategy" "$proximal_mu"; then
                                                                    echo "⏩  Skipping already completed combination"
                                                                    continue
                                                                fi
                                                                
                                                                # Clean all strategy params first
                                                                clean_strategy_params
                                                                
                                                                # Update base parameters
                                                                update_value "options.num-supernodes" "$supernodes"
                                                                update_value "num-server-rounds" "$rounds"
                                                                update_value "local-epochs" "$epochs"
                                                                update_value "num-supernodes" "$supernodes"
                                                                update_value "hidden-dim" "$hidden_dim"
                                                                update_value "latent-dim" "$latent_dim"
                                                                update_value "learning-rate" "$lr"
                                                                update_value "batch-size" "$batch"
                                                                update_value "gamma" "$gamma"
                                                                update_value "step" "$step"
                                                                update_value "fraction-fit" "$fraction"
                                                                update_value "strategy" "\"$strategy\""
                                                                update_value "patience" "$patience"
                                                                update_value "min_delta" "$min_delta"
                                                                
                                                                # Add FedProx specific parameter
                                                                add_parameter "proximal_mu" "$proximal_mu"
                                                                
                                                                echo "▶️ RUNNING: $strategy | clients=$supernodes | latent=$latent_dim | hidden=$hidden_dim | rounds=$rounds | epochs=$epochs | batch=$batch | lr=$lr | proximal_mu=$proximal_mu"
                                                                
                                                                timeout 7600 flwr run
                                                                if [ $? -eq 124 ]; then
                                                                    echo "⚠️  Timeout for this combination"
                                                                fi
                                                            done
                                                            ;;
                                                            
                                                        "fedadagrad")
                                                            for eta in "${ETA_ADAGRAD_VALUES[@]}"; do
                                                                ((COUNT++))
                                                                echo "🔁 Combination $COUNT / $TOTAL_COMBINATIONS - $strategy"
                                                                
                                                                if is_combination_done "$supernodes" "$latent_dim" "$hidden_dim" "$rounds" "$epochs" "$batch" "$lr" "$gamma" "$step" "$fraction" "$strategy" "$eta"; then
                                                                    echo "⏩  Skipping already completed combination"
                                                                    continue
                                                                fi
                                                                
                                                                # Clean all strategy params first
                                                                clean_strategy_params
                                                                
                                                                # Update base parameters
                                                                update_value "options.num-supernodes" "$supernodes"
                                                                update_value "num-server-rounds" "$rounds"
                                                                update_value "local-epochs" "$epochs"
                                                                update_value "num-supernodes" "$supernodes"
                                                                update_value "hidden-dim" "$hidden_dim"
                                                                update_value "latent-dim" "$latent_dim"
                                                                update_value "learning-rate" "$lr"
                                                                update_value "batch-size" "$batch"
                                                                update_value "gamma" "$gamma"
                                                                update_value "step" "$step"
                                                                update_value "fraction-fit" "$fraction"
                                                                update_value "strategy" "\"$strategy\""
                                                                update_value "patience" "$patience"
                                                                update_value "min_delta" "$min_delta"
                                                                
                                                                # Add FedAdagrad specific parameter
                                                                add_parameter "eta" "$eta"
                                                                
                                                                echo "▶️ RUNNING: $strategy | clients=$supernodes | latent=$latent_dim | hidden=$hidden_dim | rounds=$rounds | epochs=$epochs | batch=$batch | lr=$lr | eta=$eta"
                                                                
                                                                timeout 7600 flwr run
                                                                if [ $? -eq 124 ]; then
                                                                    echo "⚠️  Timeout for this combination"
                                                                fi
                                                            done
                                                            ;;
                                                            
                                                        "fedadam")
                                                            for eta in "${ETA_ADAM_VALUES[@]}"; do
                                                                for beta_1 in "${BETA_1_VALUES[@]}"; do
                                                                    for beta_2 in "${BETA_2_VALUES[@]}"; do
                                                                        ((COUNT++))
                                                                        echo "🔁 Combination $COUNT / $TOTAL_COMBINATIONS - $strategy"
                                                                        
                                                                        if is_combination_done "$supernodes" "$latent_dim" "$hidden_dim" "$rounds" "$epochs" "$batch" "$lr" "$gamma" "$step" "$fraction" "$strategy"; then
                                                                            echo "⏩  Skipping already completed combination"
                                                                            continue
                                                                        fi
                                                                        
                                                                        # Clean all strategy params first
                                                                        clean_strategy_params
                                                                        
                                                                        # Update base parameters
                                                                        update_value "options.num-supernodes" "$supernodes"
                                                                        update_value "num-server-rounds" "$rounds"
                                                                        update_value "local-epochs" "$epochs"
                                                                        update_value "num-supernodes" "$supernodes"
                                                                        update_value "hidden-dim" "$hidden_dim"
                                                                        update_value "latent-dim" "$latent_dim"
                                                                        update_value "learning-rate" "$lr"
                                                                        update_value "batch-size" "$batch"
                                                                        update_value "gamma" "$gamma"
                                                                        update_value "step" "$step"
                                                                        update_value "fraction-fit" "$fraction"
                                                                        update_value "strategy" "\"$strategy\""
                                                                        update_value "patience" "$patience"
                                                                        update_value "min_delta" "$min_delta"
                                                                        
                                                                        # Add FedAdam specific parameters
                                                                        add_parameter "eta" "$eta"
                                                                        add_parameter "beta_1" "$beta_1"
                                                                        add_parameter "beta_2" "$beta_2"
                                                                        
                                                                        echo "▶️ RUNNING: $strategy | clients=$supernodes | latent=$latent_dim | hidden=$hidden_dim | rounds=$rounds | epochs=$epochs | batch=$batch | lr=$lr | eta=$eta | beta1=$beta_1 | beta2=$beta_2"
                                                                        
                                                                        timeout 7600 flwr run
                                                                        if [ $? -eq 124 ]; then
                                                                            echo "⚠️  Timeout for this combination"
                                                                        fi
                                                                    done
                                                                done
                                                            done
                                                            ;;
                                                            
                                                        *)
                                                            ((COUNT++))
                                                            echo "🔁 Combination $COUNT / $TOTAL_COMBINATIONS - $strategy"
                                                            
                                                            if is_combination_done "$supernodes" "$latent_dim" "$hidden_dim" "$rounds" "$epochs" "$batch" "$lr" "$gamma" "$step" "$fraction" "$strategy"; then
                                                                echo "⏩  Skipping already completed combination"
                                                                continue
                                                            fi
                                                            
                                                            # Clean all strategy params first
                                                            clean_strategy_params
                                                            
                                                            # Update base parameters
                                                            update_value "options.num-supernodes" "$supernodes"
                                                            update_value "num-server-rounds" "$rounds"
                                                            update_value "local-epochs" "$epochs"
                                                            update_value "num-supernodes" "$supernodes"
                                                            update_value "hidden-dim" "$hidden_dim"
                                                            update_value "latent-dim" "$latent_dim"
                                                            update_value "learning-rate" "$lr"
                                                            update_value "batch-size" "$batch"
                                                            update_value "gamma" "$gamma"
                                                            update_value "step" "$step"
                                                            update_value "fraction-fit" "$fraction"
                                                            update_value "strategy" "\"$strategy\""
                                                            update_value "patience" "$patience"
                                                            update_value "min_delta" "$min_delta"
                                                            
                                                            echo "▶️ RUNNING: $strategy | clients=$supernodes | latent=$latent_dim | hidden=$hidden_dim | rounds=$rounds | epochs=$epochs | batch=$batch | lr=$lr"
                                                            
                                                            timeout 7600 flwr run
                                                            if [ $? -eq 124 ]; then
                                                                echo "⚠️  Timeout for this combination"
                                                            fi
                                                            ;;
                                                        
                                                    esac
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "✅ All experiments completed!"