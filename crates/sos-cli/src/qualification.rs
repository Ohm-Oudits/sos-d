use std::{env, fs, path::PathBuf, process::ExitCode, time::Duration};

use sos_ai::{
    qualify_humanization_model, trusted_open_model, AnthropicModelProvider,
    CloudflareModelProvider, CommandModelProvider, GroqModelProvider, HuggingFaceModelProvider,
    ModelQualificationCorpus, ModelQualificationResult, OllamaModelProvider, OpenAiModelProvider,
    CLOUDFLARE_GPT_OSS_120B, GROQ_GPT_OSS_120B, HUGGING_FACE_GPT_OSS_120B,
};

const DEFAULT_CORPUS_PROMPT: &str = "sos-model-qualification-v1";

fn anthropic_provider(api_key: String) -> Result<AnthropicModelProvider, sos_ai::AiError> {
    let provider = AnthropicModelProvider::new(api_key)?;
    match env::var("ANTHROPIC_WORKSPACE_ID") {
        Ok(workspace_id) if !workspace_id.trim().is_empty() => {
            provider.with_workspace_id(workspace_id)
        }
        _ => Ok(provider),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QualificationProvider {
    Ollama,
    Groq,
    HuggingFace,
    Cloudflare,
    OpenAi,
    Anthropic,
    Command,
}

#[derive(Debug)]
struct QualificationOptions {
    corpus: PathBuf,
    provider: QualificationProvider,
    model: String,
    ollama_url: String,
    model_command: Option<PathBuf>,
    minimum_score: f64,
    timeout: Duration,
    context_tokens: usize,
    allow_unreviewed_model: bool,
    output: Option<PathBuf>,
}

impl QualificationOptions {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let Some(corpus) = args.first() else {
            return Err("usage: sos qualify-model <corpus.json> [--provider ollama|groq|huggingface|cloudflare|openai|anthropic|command] [--model MODEL] [--ollama-url URL|--model-command PATH] [--minimum-score 0..1] [--timeout-seconds N] [--context-tokens N] [--allow-unreviewed-model] [--out PATH]".into());
        };
        let mut options = Self {
            corpus: PathBuf::from(corpus),
            provider: QualificationProvider::Ollama,
            model: "gpt-oss:20b".into(),
            ollama_url: "http://127.0.0.1:11434".into(),
            model_command: None,
            minimum_score: 0.8,
            timeout: Duration::from_secs(120),
            context_tokens: 32_768,
            allow_unreviewed_model: false,
            output: None,
        };
        let mut model_explicit = false;
        let mut index = 1;
        while index < args.len() {
            if args[index] == "--allow-unreviewed-model" {
                options.allow_unreviewed_model = true;
                index += 1;
                continue;
            }
            let flag = &args[index];
            let Some(value) = args.get(index + 1) else {
                return Err(format!("missing value for {flag}"));
            };
            match flag.as_str() {
                "--provider" => {
                    options.provider = match value.as_str() {
                        "ollama" => QualificationProvider::Ollama,
                        "groq" => QualificationProvider::Groq,
                        "huggingface" => QualificationProvider::HuggingFace,
                        "cloudflare" => QualificationProvider::Cloudflare,
                        "openai" => QualificationProvider::OpenAi,
                        "anthropic" => QualificationProvider::Anthropic,
                        "command" => QualificationProvider::Command,
                        _ => return Err(format!("unsupported qualification provider {value:?}")),
                    }
                }
                "--model" => {
                    options.model = value.clone();
                    model_explicit = true;
                }
                "--ollama-url" => options.ollama_url = value.clone(),
                "--model-command" => {
                    options.model_command = Some(PathBuf::from(value));
                    options.provider = QualificationProvider::Command;
                }
                "--minimum-score" => {
                    options.minimum_score = value
                        .parse::<f64>()
                        .map_err(|error| format!("invalid minimum score: {error}"))?;
                    if !options.minimum_score.is_finite()
                        || !(0.0..=1.0).contains(&options.minimum_score)
                    {
                        return Err("--minimum-score must be between 0 and 1".into());
                    }
                }
                "--timeout-seconds" => {
                    let seconds = value
                        .parse::<u64>()
                        .map_err(|error| format!("invalid timeout: {error}"))?;
                    if !(1..=600).contains(&seconds) {
                        return Err("--timeout-seconds must be between 1 and 600".into());
                    }
                    options.timeout = Duration::from_secs(seconds);
                }
                "--context-tokens" => {
                    options.context_tokens = value
                        .parse::<usize>()
                        .map_err(|error| format!("invalid context token count: {error}"))?;
                    if !(4_096..=131_072).contains(&options.context_tokens) {
                        return Err("--context-tokens must be between 4096 and 131072".into());
                    }
                }
                "--out" => options.output = Some(PathBuf::from(value)),
                _ => return Err(format!("unknown qualify-model option: {flag}")),
            }
            index += 2;
        }
        if !model_explicit {
            options.model = match options.provider {
                QualificationProvider::Ollama => "gpt-oss:20b".into(),
                QualificationProvider::Groq => GROQ_GPT_OSS_120B.into(),
                QualificationProvider::HuggingFace => HUGGING_FACE_GPT_OSS_120B.into(),
                QualificationProvider::Cloudflare => CLOUDFLARE_GPT_OSS_120B.into(),
                QualificationProvider::OpenAi => env::var("SOS_OPENAI_MODEL").map_err(|_| {
                    "--provider openai requires --model MODEL or SOS_OPENAI_MODEL".to_string()
                })?,
                QualificationProvider::Anthropic => {
                    env::var("SOS_ANTHROPIC_MODEL").map_err(|_| {
                        "--provider anthropic requires --model MODEL or SOS_ANTHROPIC_MODEL"
                            .to_string()
                    })?
                }
                QualificationProvider::Command => {
                    return Err("--provider command requires --model MODEL".into())
                }
            };
        }
        if options.model.trim().is_empty() {
            return Err("--model cannot be empty".into());
        }
        if options.provider == QualificationProvider::Command && options.model_command.is_none() {
            return Err("--provider command requires --model-command PATH".into());
        }
        if options.provider == QualificationProvider::Ollama
            && trusted_open_model(&options.model).is_none()
            && !options.allow_unreviewed_model
        {
            return Err(format!(
                "model {:?} is not in SOS's reviewed open-weight registry; pass --allow-unreviewed-model to benchmark it explicitly",
                options.model
            ));
        }
        Ok(options)
    }
}

pub(super) fn qualify_model_command(args: Vec<String>) -> ExitCode {
    let options = match QualificationOptions::parse(args) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(2);
        }
    };
    let corpus = match fs::read(&options.corpus)
        .map_err(|error| format!("{}: {error}", options.corpus.display()))
        .and_then(|bytes| {
            serde_json::from_slice::<ModelQualificationCorpus>(&bytes)
                .map_err(|error| error.to_string())
        }) {
        Ok(corpus) => corpus,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    let result = match options.provider {
        QualificationProvider::Ollama => qualify_humanization_model(
            OllamaModelProvider::new(&options.ollama_url)
                .with_context_tokens(options.context_tokens),
            "ollama_local_open_weights",
            &options.model,
            DEFAULT_CORPUS_PROMPT,
            &corpus,
            options.minimum_score,
        ),
        QualificationProvider::Groq => match env::var("GROQ_API_KEY") {
            Ok(api_key) => GroqModelProvider::new(api_key).and_then(|provider| {
                qualify_humanization_model(
                    provider,
                    "groq_hosted_open_weights",
                    &options.model,
                    DEFAULT_CORPUS_PROMPT,
                    &corpus,
                    options.minimum_score,
                )
            }),
            Err(_) => {
                eprintln!("error: GROQ_API_KEY is required for --provider groq");
                return ExitCode::from(2);
            }
        },
        QualificationProvider::HuggingFace => match env::var("HF_TOKEN") {
            Ok(token) => HuggingFaceModelProvider::new(token).and_then(|provider| {
                qualify_humanization_model(
                    provider,
                    "huggingface_inference_open_weights",
                    &options.model,
                    DEFAULT_CORPUS_PROMPT,
                    &corpus,
                    options.minimum_score,
                )
            }),
            Err(_) => {
                eprintln!("error: HF_TOKEN is required for --provider huggingface");
                return ExitCode::from(2);
            }
        },
        QualificationProvider::Cloudflare => {
            match (
                env::var("CLOUDFLARE_ACCOUNT_ID"),
                env::var("CLOUDFLARE_API_TOKEN"),
            ) {
                (Ok(account), Ok(token)) => {
                    CloudflareModelProvider::new(account, token).and_then(|provider| {
                        qualify_humanization_model(
                            provider,
                            "cloudflare_workers_ai_open_weights",
                            &options.model,
                            DEFAULT_CORPUS_PROMPT,
                            &corpus,
                            options.minimum_score,
                        )
                    })
                }
                _ => {
                    eprintln!("error: CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required for --provider cloudflare");
                    return ExitCode::from(2);
                }
            }
        }
        QualificationProvider::OpenAi => match env::var("OPENAI_API_KEY") {
            Ok(api_key) => OpenAiModelProvider::new(api_key).and_then(|provider| {
                qualify_humanization_model(
                    provider,
                    "openai_responses_api",
                    &options.model,
                    DEFAULT_CORPUS_PROMPT,
                    &corpus,
                    options.minimum_score,
                )
            }),
            Err(_) => {
                eprintln!("error: OPENAI_API_KEY is required for --provider openai");
                return ExitCode::from(2);
            }
        },
        QualificationProvider::Anthropic => match env::var("ANTHROPIC_API_KEY") {
            Ok(api_key) => anthropic_provider(api_key).and_then(|provider| {
                qualify_humanization_model(
                    provider,
                    "anthropic_messages_api",
                    &options.model,
                    DEFAULT_CORPUS_PROMPT,
                    &corpus,
                    options.minimum_score,
                )
            }),
            Err(_) => {
                eprintln!("error: ANTHROPIC_API_KEY is required for --provider anthropic");
                return ExitCode::from(2);
            }
        },
        QualificationProvider::Command => qualify_humanization_model(
            CommandModelProvider::new(
                options
                    .model_command
                    .as_ref()
                    .expect("command provider has a command"),
            )
            .with_timeout(options.timeout),
            "command_adapter",
            &options.model,
            DEFAULT_CORPUS_PROMPT,
            &corpus,
            options.minimum_score,
        ),
    };
    let result = match result {
        Ok(result) => result,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    if let Err(error) = emit_result(&result, options.output.as_ref()) {
        eprintln!("error: {error}");
        return ExitCode::from(1);
    }
    if result.qualified {
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "error: model qualification score {:.4} did not satisfy the corpus gate",
            result.overall_score
        );
        ExitCode::from(1)
    }
}

fn emit_result(result: &ModelQualificationResult, output: Option<&PathBuf>) -> Result<(), String> {
    let rendered = serde_json::to_vec_pretty(result).map_err(|error| error.to_string())?;
    if let Some(path) = output {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|error| format!("{}: {error}", parent.display()))?;
        }
        fs::write(path, rendered).map_err(|error| format!("{}: {error}", path.display()))?;
        println!("model_qualification={}", path.display());
    } else {
        println!("{}", String::from_utf8_lossy(&rendered));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_enforce_reviewed_local_models_and_bounded_scores() {
        let options = QualificationOptions::parse(vec!["corpus.json".into()]).unwrap();
        assert_eq!(options.model, "gpt-oss:20b");
        assert_eq!(options.minimum_score, 0.8);
        let groq = QualificationOptions::parse(vec![
            "corpus.json".into(),
            "--provider".into(),
            "groq".into(),
        ])
        .unwrap();
        assert_eq!(groq.model, GROQ_GPT_OSS_120B);
        let openai = QualificationOptions::parse(vec![
            "corpus.json".into(),
            "--provider".into(),
            "openai".into(),
            "--model".into(),
            "explicit-model".into(),
        ])
        .unwrap();
        assert_eq!(openai.model, "explicit-model");
        assert!(QualificationOptions::parse(vec![
            "corpus.json".into(),
            "--model".into(),
            "unknown:latest".into(),
        ])
        .is_err());
        assert!(QualificationOptions::parse(vec![
            "corpus.json".into(),
            "--minimum-score".into(),
            "1.1".into(),
        ])
        .is_err());
    }
}
