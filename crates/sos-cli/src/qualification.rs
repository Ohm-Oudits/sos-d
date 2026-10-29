use std::{fs, path::PathBuf, process::ExitCode, time::Duration};

use sos_ai::{
    qualify_humanization_model, trusted_open_model, CommandModelProvider, ModelQualificationCorpus,
    ModelQualificationResult, OllamaModelProvider,
};

const DEFAULT_CORPUS_PROMPT: &str = "sos-model-qualification-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QualificationProvider {
    Ollama,
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
            return Err("usage: sos qualify-model <corpus.json> [--provider ollama|command] [--model MODEL] [--ollama-url URL|--model-command PATH] [--minimum-score 0..1] [--timeout-seconds N] [--context-tokens N] [--allow-unreviewed-model] [--out PATH]".into());
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
                        "command" => QualificationProvider::Command,
                        _ => return Err(format!("unsupported qualification provider {value:?}")),
                    }
                }
                "--model" => options.model = value.clone(),
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
