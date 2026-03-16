"""
Thought Anchors Pruning Pipeline for Lean Informal Reasoning

This module implements a pipeline to reduce the length of informal reasoning
blocks in Lean proof generation while maintaining verification accuracy.

Pipeline:
1. Phase 1: Generate and filter correct Lean proofs
2. Phase 2: Compute importance scores for informal reasoning sentences
   - NLL method: Measures change in model confidence (faster)
   - KL method: Measures change in output distribution (slower, ~2-5x)
3. Phase 3: Prune low-importance sentences and create training data
"""

import re
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import logging
from pruning_common import build_chat_prompt, prune_text_by_chunks

def extract_boxed_answers(text: str) -> List[str]:
    """Extract answers enclosed in \\boxed{} from the text with improved handling
    of nested braces and complex LaTeX expressions."""
    boxed_starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    if not boxed_starts:
        return [""]
    answers = []
    for start_idx in boxed_starts:
        idx = start_idx + 7
        brace_count = 1
        answer = ""
        while idx < len(text) and brace_count > 0:
            char = text[idx]
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    break
            if brace_count > 0:
                answer += char
            idx += 1
        if answer:
            answers.append(answer)
    return answers if answers else [""]


def split_solution_into_chunks(solution_text: str) -> List[str]:
    """Split a solution into sentence-level chunks, stripping <think> tags."""
    if "<think>" in solution_text:
        solution_text = solution_text.split("<think>")[1].strip()
    if "</think>" in solution_text:
        solution_text = solution_text.split("</think>")[0].strip()

    sentence_ending_tokens = [".", "?", "!"]
    paragraph_ending_patterns = ["\n\n", "\r\n\r\n"]
    chunks = []
    current_chunk = ""
    i = 0
    while i < len(solution_text):
        current_chunk += solution_text[i]
        is_paragraph_end = any(
            i + len(p) <= len(solution_text) and solution_text[i:i + len(p)] == p
            for p in paragraph_ending_patterns
        )
        is_sentence_end = (
            i < len(solution_text) - 1
            and solution_text[i] in sentence_ending_tokens
            and solution_text[i + 1] in (" ", "\n")
        )
        if is_paragraph_end or is_sentence_end:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
        i += 1

    # Merge chunks shorter than 10 characters into neighbours
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < 10:
            if i == len(chunks) - 1:
                if i > 0:
                    chunks[i - 1] = chunks[i - 1] + " " + chunks[i]
                    chunks.pop(i)
            else:
                chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                chunks.pop(i)
            if i == 0 and len(chunks) == 1:
                break
        else:
            i += 1
    return chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LeanSample:
    """Data structure for a single Lean proof sample."""
    problem_name: str
    formal_statement: str
    full_output: str
    informal_reasoning: str
    lean_code_block: str
    proof_part: str
    is_verified: bool
    verification_status: str
    generation_metadata: Dict


@dataclass
class ChunkImportance:
    """Importance score for a chunk of informal reasoning."""
    chunk_id: int
    text: str
    start_char: int
    end_char: int
    nll_importance: float  # Importance score (NLL diff or KL divergence depending on method)
    function_tag: Optional[str] = None
    position_normalized: float = 0.0


class LeanInformalReasoningSplitter:
    """Splits model outputs into informal reasoning and formal Lean blocks."""

    # Regex pattern to extract last Lean code block
    LEAN_BLOCK_PATTERN = r"```lean4\n([\s\S]*?)\n```"

    @staticmethod
    def extract_last_lean_block(text: str) -> Optional[str]:
        """Extract the last Lean code block from text."""
        blocks = re.findall(LeanInformalReasoningSplitter.LEAN_BLOCK_PATTERN, text)
        return blocks[-1] if blocks else None

    @staticmethod
    def extract_proof_part(code: str) -> str:
        """Extract proof part (after :=) from Lean code."""
        i = code.find(":=")
        proof = code[i+2:].lstrip() if i != -1 else code
        return proof

    @staticmethod
    def split_informal_formal(full_output: str) -> Tuple[str, str]:
        """
        Split full model output into informal reasoning and formal Lean blocks.

        Args:
            full_output: Complete model output containing reasoning + Lean code

        Returns:
            (informal_reasoning, lean_code_block) tuple
            informal_reasoning: Everything before the last Lean code block
            lean_code_block: The last Lean code block content
        """
        # Find the last Lean code block
        lean_block = LeanInformalReasoningSplitter.extract_last_lean_block(full_output)

        if lean_block is None:
            # No Lean code block found, treat entire output as informal
            return full_output, ""

        # Find the start position of the last code block in the original text
        # We need to search for the actual code block marker
        lean_block_marker = f"```lean4\n{lean_block}\n```"
        last_marker_pos = full_output.rfind(lean_block_marker)

        if last_marker_pos == -1:
            # Fallback: search for just the code content
            last_marker_pos = full_output.rfind(lean_block)
            if last_marker_pos == -1:
                return full_output, lean_block

        # Everything before the marker is informal reasoning
        informal = full_output[:last_marker_pos].rstrip()

        return informal, lean_block

    @staticmethod
    def chunk_informal_reasoning(informal_text: str) -> List[Dict]:
        """
        Split informal reasoning into sentences/chunks.

        Uses the thought-anchors split_solution_into_chunks logic but adapted
        for informal reasoning (not mathematical solutions).
        """
        # Use regex to split by sentence boundaries
        # Handle bullet points, numbered lists, and paragraph breaks

        # Pattern: Split on periods followed by space/newline, or double newlines
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?:\n\s*\n)'

        # Also handle numbered lists and bullet points
        list_pattern = r'(?:\n\s*(?:\d+\.|[-*•])\s+)'

        # Combine patterns
        combined_pattern = f'({sentence_pattern}|{list_pattern})'

        # Split text
        parts = re.split(combined_pattern, informal_text)

        # Filter empty parts and reconstruct chunks
        chunks = []
        current_pos = 0

        for part in parts:
            if part and part.strip():
                text = part.strip()
                if len(text) > 3:  # Minimum chunk length
                    # Find actual position in original text
                    start_pos = informal_text.find(text, current_pos)
                    if start_pos != -1:
                        end_pos = start_pos + len(text)
                        chunks.append({
                            'text': text,
                            'start_char': start_pos,
                            'end_char': end_pos
                        })
                        current_pos = end_pos

        return chunks


class ImportanceScorer:
    """Base class for importance scorers."""

    def __init__(self, model, tokenizer, device='cuda'):
        """
        Args:
            model: HuggingFace model
            tokenizer: Corresponding tokenizer
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.model.to(device)

    def compute_chunk_importance(
        self,
        informal_chunks: List[Dict],
        informal_full: str,
        lean_code: str,
        formal_statement: str
    ) -> List[ChunkImportance]:
        """Compute importance scores for chunks. Must be implemented by subclass."""
        raise NotImplementedError

    def _construct_prompt(self, formal_statement: str, informal_reasoning: str) -> str:
        """Construct prompt matching the eval pipeline format."""
        return build_chat_prompt(
            self.tokenizer,
            formal_statement,
            assistant_prefill=informal_reasoning,
        )

    def _tokenize_with_target_indices(self, prompt: str, target: str):
        """
        Tokenize prompt+target and return token indices belonging to target text.

        Prefer character-offset alignment when available (fast tokenizers). Falls back
        to a prefix-length heuristic for slow tokenizers.
        """
        full_text = prompt + target
        boundary = len(prompt)

        try:
            enc = self.tokenizer(
                full_text,
                return_tensors='pt',
                add_special_tokens=True,
                return_offsets_mapping=True
            )
            offsets = enc.pop('offset_mapping')[0].tolist()

            target_indices = []
            for idx, (start, end) in enumerate(offsets):
                # Special tokens typically map to (0, 0)
                if start == 0 and end == 0:
                    continue
                # Any token with characters beyond prompt boundary belongs to target.
                if end > boundary:
                    target_indices.append(idx)

            return enc, target_indices
        except Exception:
            # Fallback for tokenizers without offset mapping support.
            prompt_tokens = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
            full_tokens = self.tokenizer(full_text, return_tensors='pt', add_special_tokens=True)

            prompt_ids = prompt_tokens['input_ids'][0].tolist()
            full_ids = full_tokens['input_ids'][0].tolist()

            prefix_len = 0
            for p_tok, f_tok in zip(prompt_ids, full_ids):
                if p_tok != f_tok:
                    break
                prefix_len += 1

            target_indices = list(range(prefix_len, len(full_ids)))
            return full_tokens, target_indices

    def _remove_chunk(self, text: str, chunk: Dict) -> str:
        """Remove a chunk from text by character positions."""
        start = chunk['start_char']
        end = chunk['end_char']

        # Remove chunk and clean up extra whitespace
        result = text[:start] + text[end:]

        # Clean up multiple consecutive newlines
        result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)

        return result.strip()


class NLLImportanceScorer(ImportanceScorer):
    """
    Computes importance scores for informal reasoning chunks using
    Negative Log-Likelihood (NLL) of the Lean code.

    Approach: Remove each sentence from informal reasoning, measure how much
    the model's confidence (NLL) in generating the correct Lean code changes.
    """

    def compute_nll(self, prompt: str, target: str) -> float:
        """
        Compute negative log-likelihood of target given prompt.

        Args:
            prompt: Context (informal reasoning)
            target: Target to predict (Lean code)

        Returns:
            NLL value (higher = less confident)
        """
        full_tokens, target_indices = self._tokenize_with_target_indices(prompt, target)
        input_ids = full_tokens['input_ids'].to(self.device)

        if not target_indices:
            logger.warning("No target tokens found for NLL computation; returning 0.0")
            return 0.0

        # Create labels: only target token IDs are supervised.
        labels = torch.full_like(input_ids, -100)
        idx_tensor = torch.tensor(target_indices, device=self.device)
        labels[0, idx_tensor] = input_ids[0, idx_tensor]

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss  # This is already the mean NLL

        return loss.item()

    def compute_chunk_importance(
        self,
        informal_chunks: List[Dict],
        informal_full: str,
        lean_code: str,
        formal_statement: str
    ) -> List[ChunkImportance]:
        """
        Compute importance score for each chunk in informal reasoning.

        Args:
            informal_chunks: List of chunk dicts with 'text', 'start_char', 'end_char'
            informal_full: Full informal reasoning text
            lean_code: The Lean code block (target for NLL)
            formal_statement: The formal problem statement

        Returns:
            List of ChunkImportance objects with NLL scores
        """
        # Construct the prompt with full informal reasoning
        full_prompt = self._construct_prompt(formal_statement, informal_full)

        # Compute baseline NLL with all chunks present
        baseline_nll = self.compute_nll(full_prompt, f"\n```lean4\n{lean_code}\n```")

        logger.info(f"Baseline NLL: {baseline_nll:.4f}")

        importance_scores = []

        for i, chunk in enumerate(tqdm(informal_chunks, desc="Computing importance")):
            # Create counterfactual by removing this chunk
            masked_informal = self._remove_chunk(informal_full, chunk)
            masked_prompt = self._construct_prompt(formal_statement, masked_informal)

            # Compute NLL with chunk removed
            masked_nll = self.compute_nll(masked_prompt, f"\n```lean4\n{lean_code}\n```")

            # Importance = how much NLL increases when chunk is removed
            # High positive value = chunk is important
            # Near zero or negative = chunk is redundant
            importance = masked_nll - baseline_nll

            chunk_importance = ChunkImportance(
                chunk_id=i,
                text=chunk['text'],
                start_char=chunk['start_char'],
                end_char=chunk['end_char'],
                nll_importance=importance,
                position_normalized=i / len(informal_chunks) if len(informal_chunks) > 1 else 0.0
            )

            importance_scores.append(chunk_importance)

            logger.debug(f"Chunk {i}: importance={importance:.4f}, NLL_masked={masked_nll:.4f}")

        return importance_scores


class KLImportanceScorer(ImportanceScorer):
    """
    Computes importance scores for informal reasoning chunks using
    KL Divergence of output distributions.

    Approach: Remove each sentence from informal reasoning, measure how much
    the model's output distribution over tokens changes (KL divergence).
    Higher KL = chunk has more influence on model's predictions.
    """

    def compute_kl_divergence(self, prompt: str, target: str, masked_prompt: str) -> float:
        """
        Compute KL divergence between output distributions.

        Args:
            prompt: Full context (informal reasoning with all chunks)
            target: Target to predict (Lean code)
            masked_prompt: Context with one chunk removed

        Returns:
            KL divergence D_KL(P_full || P_masked)
            Higher value = more important chunk
        """
        # Tokenize both contexts and identify target token indices robustly.
        full_tokens, full_target_indices = self._tokenize_with_target_indices(prompt, target)
        masked_tokens, masked_target_indices = self._tokenize_with_target_indices(masked_prompt, target)

        full_input_ids = full_tokens['input_ids'].to(self.device)
        masked_input_ids = masked_tokens['input_ids'].to(self.device)

        # Get logits from both contexts
        with torch.no_grad():
            # Full context logits
            full_outputs = self.model(input_ids=full_input_ids)
            full_logits = full_outputs.logits

            # Masked context logits
            masked_outputs = self.model(input_ids=masked_input_ids)
            masked_logits = masked_outputs.logits

        # Compute KL divergence for target-token predictions only.
        # Logits at position t predict token at t+1, so for target token index j
        # we compare distributions from position (j - 1).
        kl_divergence = 0.0
        num_target_tokens = 0

        full_pred_positions = [idx - 1 for idx in full_target_indices if idx > 0]
        masked_pred_positions = [idx - 1 for idx in masked_target_indices if idx > 0]
        min_target_length = min(len(full_pred_positions), len(masked_pred_positions))

        if min_target_length == 0:
            logger.warning("No aligned target-token prediction positions for KL; returning 0.0")
            return 0.0

        for i in range(min_target_length):
            full_pos = full_pred_positions[i]
            masked_pos = masked_pred_positions[i]

            # Get logits for next token prediction
            full_next_logits = full_logits[0, full_pos, :]
            masked_next_logits = masked_logits[0, masked_pos, :]

            # Convert to probabilities
            full_probs = torch.nn.functional.softmax(full_next_logits, dim=-1)
            masked_log_probs = torch.nn.functional.log_softmax(masked_next_logits, dim=-1)
            full_log_probs = torch.nn.functional.log_softmax(full_next_logits, dim=-1)

            # Compute KL divergence: sum_i P_full(i) * log(P_full(i) / P_masked(i))
            # = sum_i P_full(i) * (log P_full(i) - log P_masked(i))
            kl = (full_probs * (full_log_probs - masked_log_probs)).sum()

            kl_divergence += kl.item()
            num_target_tokens += 1

        # Return average KL divergence per token
        return kl_divergence / num_target_tokens if num_target_tokens > 0 else 0.0

    def compute_chunk_importance(
        self,
        informal_chunks: List[Dict],
        informal_full: str,
        lean_code: str,
        formal_statement: str
    ) -> List[ChunkImportance]:
        """
        Compute importance score for each chunk in informal reasoning.

        Args:
            informal_chunks: List of chunk dicts with 'text', 'start_char', 'end_char'
            informal_full: Full informal reasoning text
            lean_code: The Lean code block (target for KL computation)
            formal_statement: The formal problem statement

        Returns:
            List of ChunkImportance objects with KL divergence scores
        """
        # Construct the prompt with full informal reasoning
        full_prompt = self._construct_prompt(formal_statement, informal_full)
        target = f"\n```lean4\n{lean_code}\n```"

        logger.info(f"Computing KL-based importance scores...")

        importance_scores = []

        for i, chunk in enumerate(tqdm(informal_chunks, desc="Computing KL importance")):
            # Create counterfactual by removing this chunk
            masked_informal = self._remove_chunk(informal_full, chunk)
            masked_prompt = self._construct_prompt(formal_statement, masked_informal)

            # Compute KL divergence
            kl_div = self.compute_kl_divergence(full_prompt, target, masked_prompt)

            # Importance = KL divergence when chunk is removed
            # High positive value = chunk has strong influence on output distribution
            chunk_importance = ChunkImportance(
                chunk_id=i,
                text=chunk['text'],
                start_char=chunk['start_char'],
                end_char=chunk['end_char'],
                nll_importance=kl_div,  # Store in nll_importance field for compatibility
                position_normalized=i / len(informal_chunks) if len(informal_chunks) > 1 else 0.0
            )

            importance_scores.append(chunk_importance)

            logger.debug(f"Chunk {i}: KL importance={kl_div:.4f}")

        return importance_scores


class LeanPruningPipeline:
    """
    Main pipeline for pruning informal reasoning in Lean proofs.

    Pipeline stages:
    1. Load verified samples (Phase 1: Gold Mine)
    2. Split informal/formal blocks
    3. Chunk informal reasoning
    4. Compute importance scores (Phase 2: Pruning Engine)
    5. Prune low-importance chunks
    6. Generate training data
    """

    def __init__(
        self,
        model,
        tokenizer,
        device='cuda',
        importance_threshold: float = 0.0,
        use_function_tags: bool = False,
        importance_method: str = 'nll'
    ):
        """
        Args:
            model: Model for importance computation
            tokenizer: Corresponding tokenizer
            device: Computation device
            importance_threshold: Chunks with importance < threshold are pruned
            use_function_tags: Whether to use LLM labeling for function tags
            importance_method: Method for computing importance ('nll' or 'kl')
        """
        self.splitter = LeanInformalReasoningSplitter()

        # Create appropriate scorer based on method
        if importance_method == 'nll':
            self.scorer = NLLImportanceScorer(model, tokenizer, device)
        elif importance_method == 'kl':
            self.scorer = KLImportanceScorer(model, tokenizer, device)
        else:
            raise ValueError(f"Unknown importance method: {importance_method}. Use 'nll' or 'kl'")

        self.importance_threshold = importance_threshold
        self.use_function_tags = use_function_tags
        self.importance_method = importance_method

    def process_verified_samples(
        self,
        samples: List[LeanSample],
        output_dir: Path
    ) -> List[Dict]:
        """
        Process a list of verified samples through the full pipeline.

        Args:
            samples: List of LeanSample objects (must have is_verified=True)
            output_dir: Directory to save results

        Returns:
            List of pruned sample dicts ready for training
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pruned_samples = []

        for sample in tqdm(samples, desc="Processing samples"):
            if not sample.is_verified:
                logger.warning(f"Skipping unverified sample: {sample.problem_name}")
                continue

            try:
                pruned_sample = self._process_single_sample(sample, output_dir)
                pruned_samples.append(pruned_sample)
            except Exception as e:
                logger.error(f"Error processing {sample.problem_name}: {e}")
                continue

        # Save all results
        results_file = output_dir / "pruned_samples.json"
        with open(results_file, 'w') as f:
            json.dump(pruned_samples, f, indent=2)

        logger.info(f"Saved {len(pruned_samples)} pruned samples to {results_file}")

        return pruned_samples

    def _process_single_sample(self, sample: LeanSample, output_dir: Path) -> Dict:
        """Process a single sample through chunking, scoring, and pruning."""
        logger.info(f"Processing: {sample.problem_name}")

        # 1. Split informal/formal
        informal, lean_code = self.splitter.split_informal_formal(sample.full_output)

        # 2. Chunk informal reasoning
        chunks = self.splitter.chunk_informal_reasoning(informal)
        logger.info(f"  Chunked into {len(chunks)} sentences")

        if len(chunks) == 0:
            logger.warning(f"  No chunks found for {sample.problem_name}")
            return self._create_output_dict(sample, [], informal, lean_code)

        # 3. Compute importance scores
        importance_scores = self.scorer.compute_chunk_importance(
            chunks,
            informal,
            lean_code,
            sample.formal_statement
        )

        # 4. Optional: Label with function tags
        if self.use_function_tags:
            importance_scores = self._add_function_tags(importance_scores)

        # 5. Prune low-importance chunks
        pruned_informal = self._prune_chunks(informal, importance_scores)

        # Calculate reduction
        original_length = len(informal)
        pruned_length = len(pruned_informal)
        reduction_pct = (1 - pruned_length / original_length) * 100 if original_length > 0 else 0

        logger.info(f"  Pruned: {original_length} → {pruned_length} chars ({reduction_pct:.1f}% reduction)")

        # 6. Create output
        output = self._create_output_dict(
            sample, importance_scores, pruned_informal, lean_code
        )
        output['reduction_percentage'] = reduction_pct

        # Save individual sample
        sample_file = output_dir / f"{sample.problem_name}_pruned.json"
        with open(sample_file, 'w') as f:
            json.dump(output, f, indent=2)

        return output

    def _prune_chunks(self, informal: str, importance_scores: List[ChunkImportance]) -> str:
        """
        Remove chunks with importance below threshold.

        Keeps chunks in order, removes low-importance ones.
        """
        chunks_to_remove = [
            score for score in importance_scores if score.nll_importance < self.importance_threshold
        ]
        result = prune_text_by_chunks(informal, chunks_to_remove)
        pruned_count = len(chunks_to_remove)

        logger.info(f"  Pruned {pruned_count}/{len(importance_scores)} chunks")

        return result

    def _add_function_tags(self, importance_scores: List[ChunkImportance]) -> List[ChunkImportance]:
        """
        Add function tags to chunks using LLM labeling.

        Note: This would require an OpenAI API call in production.
        For now, we'll implement a heuristic version.
        """
        # TODO: Implement LLM-based labeling using prompts.py
        # For now, use simple heuristics

        for score in importance_scores:
            text_lower = score.text.lower()

            # Simple heuristic rules
            if any(word in text_lower for word in ['let', 'suppose', 'assume', 'given']):
                score.function_tag = 'Setup'
            elif any(word in text_lower for word in ['plan', 'strategy', 'approach', 'first', 'then']):
                score.function_tag = 'PlanGeneration'
            elif any(word in text_lower for word in ['calculate', 'compute', 'evaluate', '=']):
                score.function_tag = 'ActiveComputation'
            elif any(word in text_lower for word in ['therefore', 'thus', 'so', 'hence', 'conclude']):
                score.function_tag = 'FinalAnswer'
            elif any(word in text_lower for word in ['check', 'verify', 'confirm']):
                score.function_tag = 'Verification'
            else:
                score.function_tag = 'Other'

        return importance_scores

    def _create_output_dict(
        self,
        sample: LeanSample,
        importance_scores: List[ChunkImportance],
        pruned_informal: str,
        lean_code: str
    ) -> Dict:
        """Create output dictionary for training data."""
        return {
            'problem_name': sample.problem_name,
            'formal_statement': sample.formal_statement,
            'original_output': sample.full_output,
            'original_informal': sample.informal_reasoning,
            'pruned_informal': pruned_informal,
            'lean_code': lean_code,
            'proof_part': sample.proof_part,
            'importance_scores': [
                {
                    'chunk_id': s.chunk_id,
                    'text': s.text,
                    'start_char': s.start_char,
                    'end_char': s.end_char,
                    'nll_importance': s.nll_importance,
                    'function_tag': s.function_tag,
                    'position_normalized': s.position_normalized
                }
                for s in importance_scores
            ],
            'metadata': sample.generation_metadata
        }


def load_verified_samples_from_results(
    results_file: Path,
    outputs_dir: Path,
    formal_statements: Dict[str, str]
) -> List[LeanSample]:
    """
    Load verified samples from eval_baseline.py results.

    Args:
        results_file: Path to results JSON from eval_baseline.py
        outputs_dir: Directory containing saved model outputs
        formal_statements: Dict mapping problem_name -> formal statement

    Returns:
        List of LeanSample objects with verified proofs
    """
    with open(results_file) as f:
        results = json.load(f)

    samples = []

    # Find the seed key (e.g., "seed_1")
    seed_keys = [k for k in results.keys() if k.startswith('seed_') and not k.endswith('_metrics')]

    for seed_key in seed_keys:
        seed_results = results[seed_key]

        for problem_key, correct_count in seed_results.items():
            if not problem_key.startswith('correct_'):
                continue

            problem_idx = int(problem_key.split('_')[1])

            if correct_count > 0:
                # Load the corresponding outputs
                # This assumes you've saved outputs during evaluation
                # You'll need to modify eval_baseline.py to save full outputs

                # For now, placeholder
                logger.warning(
                    "load_verified_samples_from_results requires saved model outputs. "
                    "You need to modify eval_baseline.py to save full outputs for each sample."
                )

    return samples


# Example usage function
def example_pipeline():
    """
    Example of how to use the pruning pipeline.

    You'll need to:
    1. Modify eval_baseline.py to save full model outputs (not just verification status)
    2. Run evaluation to generate verified samples
    3. Load those samples and process through this pipeline
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model for NLL computation
    model_path = "your_model_path"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Create pipeline
    pipeline = LeanPruningPipeline(
        model=model,
        tokenizer=tokenizer,
        device='cuda',
        importance_threshold=0.0,  # Prune chunks with importance <= 0
        use_function_tags=False
    )

    # Load verified samples
    # samples = load_verified_samples_from_results(...)

    # Process samples
    # pruned_samples = pipeline.process_verified_samples(samples, output_dir=Path("pruned_data"))

    print("Pipeline setup complete. See docstrings for usage.")


if __name__ == "__main__":
    example_pipeline()
