
import re
import logging
from typing import List, Tuple, Optional, Set, Dict
from difflib import SequenceMatcher
from collections import defaultdict
import json
from pathlib import Path

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    print("WARNING: pyspellchecker not installed. Install with: pip install pyspellchecker")

try:
    import nltk
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("WARNING: nltk not installed. Install with: pip install nltk")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Query processor with typo correction for academic search.
    
    This system uses a multi-level approach to typo correction:
    
    1. **Common typos dictionary** - manual dictionary of typical errors in NLP/ML domain
    2. **NLP-specific vocabulary** - terms that should NOT be corrected
    3. **Common English dictionary** - preloaded dictionary of common words (~30k words)
    4. **WordNet** (optional) - for checking word existence
    
    Атрибуты:
        spell (SpellChecker): Main spell checker with preloaded dictionary
        min_confidence (float): Minimum confidence for applying correction
        max_corrections (int): Maximum number of corrections per query
        _nlp_vocabulary (set): NLP/ML specialized terms
        _common_typos (dict): Dictionary of known typos → correct spelling
        _correction_cache (dict): Cache for speeding up repeated corrections
    """
    
    def __init__(
        self,
        custom_vocabulary: Optional[List[str]] = None,
        min_confidence: float = 0.7,
        max_corrections: int = 3,
        enable_cache: bool = True,
        dictionary_path: Optional[str] = None
    ):
        """
        Initialize query processor.
        
        Args:
            custom_vocabulary: Additional terms for the dictionary
            min_confidence: Confidence threshold (0.0-1.0). Corrections with lower
                          confidence are ignored
            max_corrections: Limit on the number of corrections in one query
            enable_cache: Enable correction cache
            dictionary_path: Path to the user dictionary (text file)
        """
        self.min_confidence = min_confidence
        self.max_corrections = max_corrections
        self.enable_cache = enable_cache
        self._correction_cache: Dict[str, Tuple[str, float]] = {}
        
        if SPELLCHECKER_AVAILABLE:
            self.spell = SpellChecker()
            try:
                if hasattr(self.spell.word_frequency, "unique_words"):
                    num_words = self.spell.word_frequency.unique_words  # type: ignore[attr-defined]
                else:
                    num_words = len(list(self.spell.word_frequency.words()))  # type: ignore[arg-type]
            except Exception:
                num_words = "unknown"
            logger.info(f"SpellChecker initialized (words in dictionary: {num_words})")
        else:
            self.spell = None
            logger.warning("SpellChecker not available, using fallback mode")
        
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('corpora/wordnet')
                self.use_wordnet = True
                logger.info("WordNet available for additional validation")
            except LookupError:
                logger.info("Downloading WordNet...")
                nltk.download('wordnet')
                self.use_wordnet = True
        else:
            self.use_wordnet = False
        
        self._nlp_vocabulary = {
            'transformer', 'transformers', 'bert', 'roberta', 'xlnet', 'albert',
            'electra', 'gpt', 'gpt2', 'gpt3', 'gpt4', 't5', 'bart', 'mbart',
            'xlm', 'xlmr', 'distilbert', 'deberta', 'longformer', 'reformer',
            'funnel', 'convbert', 'layoutlm', 'tapas', 'flaubert', 'camembert',
            
            'lstm', 'gru', 'rnn', 'cnn', 'gcn', 'gnn', 'gan', 'vae', 'ae',
            'mlp', 'ffn', 'resnet', 'densenet', 'efficientnet', 'vgg',
            
            'embedding', 'embeddings', 'encoder', 'decoder', 'attention',
            'multihead', 'crossattention', 'selfattention', 'positional',
            'tokenizer', 'tokenization', 'subword', 'wordpiece', 'sentencepiece',
            'bpe', 'dropout', 'layernorm', 'batchnorm', 'normalization',
            
            'pretrained', 'finetuning', 'finetune', 'finetuned', 'pretraining',
            'downstream', 'upstream', 'zeroshot', 'fewshot', 'multitask',
            'transfer', 'learning', 'supervised', 'unsupervised', 'semisupervised',
            'reinforcement', 'metalearning', 'continual', 'federated',
            
            'classification', 'generation', 'summarization', 'translation',
            'parsing', 'tagging', 'ner', 'pos', 'srl', 'coref', 'coreference',
            'qa', 'qg', 'nli', 'sts', 'paraphrase', 'entailment', 'sentiment',
            'emotion', 'stance', 'factchecking', 'textual', 'similarity',
            
            'semantic', 'syntactic', 'morphological', 'phonological', 'pragmatic',
            'lexical', 'discourse', 'anaphora', 'cataphora', 'deixis',
            'polysemy', 'synonymy', 'antonymy', 'hyponymy', 'hypernymy',
            
            'dataset', 'corpus', 'corpora', 'benchmark', 'leaderboard',
            'train', 'validation', 'dev', 'test', 'split', 'fold',
            'augmentation', 'preprocessing', 'postprocessing',
            
            'accuracy', 'precision', 'recall', 'f1', 'bleu', 'rouge', 'meteor',
            'bertscore', 'bleurt', 'comet', 'chrff', 'ter', 'wer', 'cer',
            'perplexity', 'loss', 'crossentropy', 'mse', 'mae', 'rmse',
            
            'acl', 'naacl', 'eacl', 'aacl', 'emnlp', 'coling', 'lrec',
            'conll', 'tacl', 'cl', 'aaai', 'ijcai', 'icml', 'neurips',
            'iclr', 'cvpr', 'eccv', 'iccv', 'acmmm', 'sigir', 'sigkdd',
            
            'nlp', 'nlu', 'nlg', 'mt', 'ir', 'kg', 'kge', 'ie', 're',
            'api', 'sdk', 'gpu', 'cpu', 'tpu', 'cuda', 'onnx', 'tensorrt',
            'pytorch', 'tensorflow', 'jax', 'keras', 'huggingface', 'spacy',
            'nltk', 'stanford', 'allennlp', 'fairseq', 'transformers',
            'tokenizers', 'datasets', 'wandb', 'mlflow', 'tensorboard',
            'arxiv', 'openreview', 'github', 'sota', 'baseline', 'ablation'
        }
        
        self._common_typos = {}
        
        if custom_vocabulary:
            self._nlp_vocabulary.update(word.lower() for word in custom_vocabulary)
            logger.info(f"Added {len(custom_vocabulary)} custom vocabulary terms")
        
        if dictionary_path and Path(dictionary_path).exists():
            self._load_custom_dictionary(dictionary_path)
        
        if self.spell:
            self.spell.word_frequency.load_words(self._nlp_vocabulary)
            logger.info(f"Updated spellchecker with {len(self._nlp_vocabulary)} NLP terms")
        
        self._stats = {
            'total_queries': 0,
            'corrected_queries': 0,
            'total_corrections': 0,
            'cache_hits': 0,
            'typo_frequencies': defaultdict(int)
        }
    
    def _load_custom_dictionary(self, path: str) -> None:
        """Load additional dictionary from file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                words = {line.strip().lower() for line in f if line.strip()}
                self._nlp_vocabulary.update(words)
                logger.info(f"Loaded {len(words)} words from {path}")
        except Exception as e:
            logger.error(f"Failed to load dictionary from {path}: {e}")
    
    def process_query(self, query: str) -> Tuple[str, List[str], float]:
        """
        Process and correct user query.
        
        Args:
            query: Original user query
            
        Returns:
            Tuple containing:
            - corrected_query: Corrected query
            - corrections: List of applied corrections in format "was → became"
            - confidence: Confidence in corrections (0.0-1.0)
        """
        self._stats['total_queries'] += 1
        
        cleaned_query = self._clean_query(query)
        
        corrected_query, corrections = self._correct_typos(cleaned_query)
        
        confidence = self._calculate_confidence(query, corrected_query, corrections)
        
        if corrections:
            self._stats['corrected_queries'] += 1
            self._stats['total_corrections'] += len(corrections)
            for correction in corrections:
                typo = correction.split(' → ')[0]
                self._stats['typo_frequencies'][typo] += 1
        
        logger.info(f"Query processing: '{query}' -> '{corrected_query}' "
                   f"(confidence: {confidence:.2f}, corrections: {len(corrections)})")
        
        return corrected_query, corrections, confidence
    
    def _clean_query(self, query: str) -> str:
        """
        Clean and normalize query.
        """
        query = ' '.join(query.split())
        
        query = re.sub(r'\s+([?.!,])', r'\1', query)
        query = re.sub(r'([?.!,])\s*', r'\1 ', query)
        
        if query.count('"') % 2 != 0:
            query = query.replace('"', '')
        
        return query.strip()
    
    def _correct_typos(self, query: str) -> Tuple[str, List[str]]:
        """
        Main typo correction logic.
        """
        words = query.lower().split()
        corrected_words = []
        corrections = []
        correction_count = 0
        
        for i, word in enumerate(words):
            if correction_count >= self.max_corrections:
                corrected_words.append(word)
                continue
            
            if word in self._nlp_vocabulary:
                corrected_words.append(word)
                continue
            
            if len(word) <= 2:
                corrected_words.append(word)
                continue
            
            if self.enable_cache and word in self._correction_cache:
                cached_correction, cached_confidence = self._correction_cache[word]
                if cached_confidence >= self.min_confidence:
                    corrected_words.append(cached_correction)
                    if cached_correction != word:
                        corrections.append(f"{word} → {cached_correction}")
                        correction_count += 1
                    self._stats['cache_hits'] += 1
                else:
                    corrected_words.append(word)
                continue
            
            correction = self._get_spell_correction(word)
            if correction and correction != word:
                similarity = SequenceMatcher(None, word, correction).ratio()
                
                if similarity >= self.min_confidence:
                    corrected_words.append(correction)
                    corrections.append(f"{word} → {correction}")
                    correction_count += 1

                    if self.enable_cache:
                        self._correction_cache[word] = (correction, similarity)
                else:
                    corrected_words.append(word)
                    if self.enable_cache:
                        self._correction_cache[word] = (word, similarity)
            else:
                corrected_words.append(word)
                if self.enable_cache:
                    self._correction_cache[word] = (word, 1.0)
        
        corrected_query = self._restore_case(corrected_words, query.split())
        
        return corrected_query, corrections
    
    def _get_spell_correction(self, word: str) -> Optional[str]:
        """
        Get correction from spell checker with additional validation.
        """
        if not self.spell:
            return None
        
        correction = self.spell.correction(word)
        
        if self.use_wordnet and correction and correction != word:
            if wordnet.synsets(word):
                return word
            else:
                candidates = self.spell.candidates(word)
                if candidates:
                    for candidate in candidates:
                        if wordnet.synsets(candidate):
                            return candidate
        
        return correction
    
    def _restore_case(self, corrected_words: List[str], original_words: List[str]) -> str:
        """
        Restore case and formatting from the original query.
        """
        result = []
        
        acronyms = {
            'nlp': 'NLP', 'bert': 'BERT', 'gpt': 'GPT', 'lstm': 'LSTM',
            'rnn': 'RNN', 'cnn': 'CNN', 'gan': 'GAN', 'vae': 'VAE',
            'ner': 'NER', 'pos': 'POS', 'srl': 'SRL', 'qa': 'QA',
            'nli': 'NLI', 'mt': 'MT', 'ir': 'IR', 'kg': 'KG', 'ie': 'IE',
            'acl': 'ACL', 'emnlp': 'EMNLP', 'naacl': 'NAACL', 
            'coling': 'COLING', 'lrec': 'LREC', 'conll': 'CoNLL',
            'aaai': 'AAAI', 'ijcai': 'IJCAI', 'icml': 'ICML',
            'neurips': 'NeurIPS', 'iclr': 'ICLR', 'cvpr': 'CVPR',
            'sota': 'SOTA', 'bleu': 'BLEU', 'rouge': 'ROUGE', 'f1': 'F1',
            'api': 'API', 'sdk': 'SDK', 'gpu': 'GPU', 'cpu': 'CPU',
            'tpu': 'TPU', 'cuda': 'CUDA', 'onnx': 'ONNX'
        }
        
        for i, word in enumerate(corrected_words):
            if word in acronyms:
                result.append(acronyms[word])
            elif i < len(original_words):
                if original_words[i][0].isupper():
                    result.append(word.capitalize())
                elif original_words[i].isupper():
                    result.append(word.upper())
                else:
                    result.append(word)
            else:
                result.append(word)
        
        return ' '.join(result)
    
    def _calculate_confidence(self, original: str, corrected: str, corrections: List[str]) -> float:
        """
        Calculate confidence in corrections.
        
        Factors:
        - Similarity between original and corrected string
        - Number of corrections
        """
        if not corrections:
            return 1.0
        
        similarity = SequenceMatcher(None, original.lower(), corrected.lower()).ratio()
        
        correction_penalty = len(corrections) * 0.05
        
        confidence = min(1.0, max(0.0, similarity - correction_penalty))
        
        return confidence
    
    def suggest_alternatives(self, query: str, top_k: int = 3) -> List[str]:
        """
        Suggest alternative query variants.
        
        """
        alternatives = []
        words = query.lower().split()
        
        for i, word in enumerate(words):
            if word in self._nlp_vocabulary or len(word) <= 2:
                continue
            
            if self.spell:
                candidates = self.spell.candidates(word)
                if candidates and word not in candidates:
                    for candidate in list(candidates)[:2]:
                        alt_words = words.copy()
                        alt_words[i] = candidate
                        alt_query = ' '.join(alt_words)
                        if alt_query != query.lower():
                            alternatives.append(self._restore_case(
                                alt_words, query.split()
                            ))
        
        seen = set()
        unique_alternatives = []
        for alt in alternatives:
            if alt not in seen:
                seen.add(alt)
                unique_alternatives.append(alt)
        
        return unique_alternatives[:top_k]
    
    def get_statistics(self) -> Dict:
        """Get usage statistics."""
        return {
            'total_queries': self._stats['total_queries'],
            'corrected_queries': self._stats['corrected_queries'],
            'correction_rate': (self._stats['corrected_queries'] / 
                              max(1, self._stats['total_queries'])),
            'total_corrections': self._stats['total_corrections'],
            'cache_hits': self._stats['cache_hits'],
            'cache_size': len(self._correction_cache),
            'most_common_typos': dict(sorted(
                self._stats['typo_frequencies'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
        }
    
    def save_cache(self, path: str) -> None:
        """Save correction cache for future use."""
        cache_data = {
            'cache': self._correction_cache,
            'stats': self._stats,
            'typo_frequencies': dict(self._stats['typo_frequencies'])
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Cache saved to {path}")
    
    def load_cache(self, path: str) -> None:
        """Load correction cache."""
        if Path(path).exists():
            with open(path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                self._correction_cache = cache_data.get('cache', {})
                self._correction_cache = {
                    k: tuple(v) for k, v in self._correction_cache.items()
                }
                logger.info(f"Cache loaded from {path} ({len(self._correction_cache)} entries)")


def preprocess_query(
    query: str,
    processor: Optional[QueryProcessor] = None
) -> Tuple[str, float]:
    """
    Simple function for integration into existing code.
    
    Args:
        query: User query
        processor: QueryProcessor instance (created automatically if None)
        
    Returns:
        Tuple (corrected_query, confidence)
    """
    if processor is None:
        processor = QueryProcessor()
    
    corrected_query, corrections, confidence = processor.process_query(query)
    
    if corrections:
        logger.info(f"Query corrections applied: {', '.join(corrections)}")
    
    return corrected_query, confidence


def demo():
    """Demo of all typo correction capabilities."""
    
    print("=" * 80)
    print("TYPO CORRECTION SYSTEM FOR ACADEMIC SEARCH")
    print("=" * 80)
    
    processor = QueryProcessor(
        min_confidence=0.7,
        max_corrections=3,
        enable_cache=True
    )
    
    test_queries = [
        "tansformer attenton mechansim",
        "embedings for nlp",
        "nueral netwrok architechture",
        
        "bert modle for clasification",
        "gpt3 langauge generation",
        
        "sumarization with transformet",
        "sentement analysis dataset",
        
        "evaluaiton metrics blue and rough",
        "f-1 score and acuracy",
        
        "emnpl 2024 paper",
        "nacaal workshop on mt",
        
        "preprocesing for tokenizaton",
        "benchamrk performace comparison",
        
        "transformer attention mechanism",
        "BERT embeddings visualization",
        "neural machine translation",
        
        "transfomer-based aproach for nlp",
        "multi-head attenton in bert",
    ]
    
    print("\nTEST CASES:")
    print("-" * 80)
    
    for i, query in enumerate(test_queries, 1):
        corrected, corrections, confidence = processor.process_query(query)
        
        print(f"\n{i}. Original query:  {query}")
        print(f"   Corrected:        {corrected}")
        
        if corrections:
            print(f"   Corrections:      {' | '.join(corrections)}")
        else:
            print("   Corrections:      [not required]")
        
        print(f"   Confidence:       {confidence:.1%}")
        
        if confidence < 0.8 and corrections:
            alternatives = processor.suggest_alternatives(query)
            if alternatives:
                print(f"   Alternatives:     {' | '.join(alternatives[:2])}")
    
    print("\n\nSTATISTICS:")
    print("-" * 80)
    stats = processor.get_statistics()
    print(f"Total queries:            {stats['total_queries']}")
    print(f"Corrected queries:        {stats['corrected_queries']}")
    print(f"Correction rate:          {stats['correction_rate']:.1%}")
    print(f"Total corrections:        {stats['total_corrections']}")
    print(f"Cache hits:               {stats['cache_hits']}")
    print(f"Cache size:               {stats['cache_size']}")
    
    if stats['most_common_typos']:
        print("\nTop-5 frequent typos:")
        for typo, count in list(stats['most_common_typos'].items())[:5]:
            print(f"  - '{typo}': {count} times")
    
    print("\n\nINTERACTIVE MODE")
    print("-" * 80)
    print("Enter queries (type 'exit' to quit)")
    print("Commands: 'stats' - show stats, 'cache' - save cache")
    print("-" * 80)
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            elif user_input.lower() == 'stats':
                stats = processor.get_statistics()
                print(f"\nProcessed queries: {stats['total_queries']}")
                print(f"Correction rate: {stats['correction_rate']:.1%}")
                continue
            elif user_input.lower() == 'cache':
                processor.save_cache('query_corrections_cache.json')
                print("Cache saved to query_corrections_cache.json")
                continue
            
            if not user_input:
                continue
            
            corrected, corrections, confidence = processor.process_query(user_input)
            
            print(f"\nResult: {corrected}")
            
            if corrections:
                print(f"  Corrections: {' | '.join(corrections)}")
                print(f"  Confidence: {confidence:.1%}")
                
                if confidence < 0.8:
                    alternatives = processor.suggest_alternatives(user_input)
                    if alternatives:
                        print(f"  Alternatives: {' | '.join(alternatives)}")
            else:
                print("  [No corrections required]")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
    
    processor.save_cache('query_corrections_cache.json')
    print("\nCache saved for future use")


if __name__ == "__main__":
    print("Dependency check...")
    if not SPELLCHECKER_AVAILABLE:
        print("WARNING: pyspellchecker not installed. Install with: pip install pyspellchecker")
    if not NLTK_AVAILABLE:
        print("WARNING: nltk not installed. Install with: pip install nltk")
    
    if SPELLCHECKER_AVAILABLE:
        print("pyspellchecker available")
    if NLTK_AVAILABLE:
        print("nltk available")
    
    print()
    
    demo()