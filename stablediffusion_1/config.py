import argparse
from dataclasses import dataclass


@dataclass
class TrainingConfig:

    model_name: str = "CompVis/stable-diffusion-v1-4"
    cache_dir: str = "./cache/"

    dataset: str = "harmful_to_safe"

    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    lr_encoder: float = 1e-4
    lr_disc: float = 1e-3

    num_epochs: int = 10
    batch_size: int = 16
    lambda_preserve: float = 1.0
    lambda_adv: float = 0.5

    ema_decay: float = 0.999
    use_ema: bool = True

    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    generation_seed: int = 42
    generate_every_n_epochs: int = 1
    num_sample_prompts: int = 5

    device: str = "cuda"

    output_dir: str = "./outputs"
    save_model_dir: str = "./saved_models"

    use_wandb: bool = False
    wandb_project: str = "adversarial-unlearning"
    wandb_run_name: str = "run"
    wandb_entity: str = None

    use_fixed_target_emb: bool = False
    pooled_embed: bool = False

    preservation_type: str = "token_cosine"
    kl_temperature: float = 1.0


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="SD1.4 Unlearning Training")

    parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--cache_dir", type=str, default="./cache/")

    parser.add_argument("--use_lora", action="store_true", default=False, help="Use LoRA (default: full fine-tuning)")
    parser.add_argument("--no_lora", dest="use_lora", action="store_false", help="Disable LoRA (full fine-tuning, default)")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    parser.add_argument("--lr_encoder", type=float, default=1e-4, help="Learning rate for encoder")
    parser.add_argument("--lr_disc", type=float, default=1e-3, help="Learning rate for discriminator")

    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lambda_preserve", type=float, default=1.0, help="Weight for preservation loss")
    parser.add_argument("--lambda_adv", type=float, default=0.5, help="Weight for adversarial loss")
    parser.add_argument(
        "--dataset",
        type=str,
        default="harmful_to_safe",
        choices=[
            "harmful_to_safe",
            "vangogh_style_to_safe",
            "kelly_mckernan_style_to_safe_prompts",
        ],
        help="Dataset choice",
    )
    parser.add_argument("--use_fixed_target_emb", action="store_true", help="Use fixed target embeddings")
    parser.add_argument("--pooled_embed", action="store_true", help="Use pooled embeddings instead of token-level")
    parser.add_argument(
        "--preservation_type",
        type=str,
        default="token_cosine",
        choices=["pooled_l2", "token_l2", "token_cosine", "token_kl"],
        help="Type of preservation loss: pooled_l2, token_l2, token_cosine (default), or token_kl",
    )
    parser.add_argument("--kl_temperature", type=float, default=1.0, help="Temperature for token_kl softmax")

    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")
    parser.add_argument("--use_ema", action="store_true", default=True, help="Use EMA for training")
    parser.add_argument("--no_ema", dest="use_ema", action="store_false", help="Disable EMA")

    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--generation_seed", type=int, default=42)
    parser.add_argument("--generate_every_n_epochs", type=int, default=1)
    parser.add_argument("--num_sample_prompts", type=int, default=5)

    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="adversarial-unlearning", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default="run", help="WandB run name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity/username")

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--save_model_dir", type=str, default="./saved_models")

    args = parser.parse_args()

    return TrainingConfig(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        dataset=args.dataset,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lr_encoder=args.lr_encoder,
        lr_disc=args.lr_disc,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lambda_preserve=args.lambda_preserve,
        lambda_adv=args.lambda_adv,
        ema_decay=args.ema_decay,
        use_ema=args.use_ema,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generation_seed=args.generation_seed,
        generate_every_n_epochs=args.generate_every_n_epochs,
        num_sample_prompts=args.num_sample_prompts,
        device=args.device,
        output_dir=args.output_dir,
        save_model_dir=args.save_model_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        pooled_embed=args.pooled_embed,
        use_fixed_target_emb=args.use_fixed_target_emb,
        preservation_type=args.preservation_type,
        kl_temperature=args.kl_temperature,
    )
