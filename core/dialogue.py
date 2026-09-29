"""
ClarifySign - Dialogue State Management Module
Tracks multi-turn conversation state, semantic concept memory with temporal decay,
active domain focus, and clarification resolution.
"""

from dataclasses import dataclass, field
import time
from typing import List, Dict, Optional, Any


@dataclass
class Turn:
    """Represents a single conversational turn in the dialogue."""
    speaker: str              # 'customer' | 'shopkeeper' | 'system'
    raw: str                  # Raw gesture label or question/response text
    semantic: str = ""        # Disambiguated semantic concept
    language: str = "English" # Output language
    was_clarified: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            "speaker": self.speaker,
            "raw": self.raw,
            "semantic": self.semantic,
            "language": self.language,
            "was_clarified": self.was_clarified,
            "time_str": time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        }


@dataclass
class DialogueState:
    """
    Maintains compact conversation state:
    - turn history
    - semantic concept memory with temporal decay
    - pending clarification question
    - resolved concepts list
    - active conversation domain
    """
    turns: List[Turn] = field(default_factory=list)
    context: Dict[str, float] = field(default_factory=dict)
    pending_question: Optional[Any] = field(default=None)
    pending_options: List[str] = field(default_factory=list)
    pending_candidate_keys: List[str] = field(default_factory=list)
    resolved_concepts: List[str] = field(default_factory=list)
    active_domain: str = "general"
    decay_factor: float = 0.85

    @property
    def pending(self):
        """Returns True if there is a pending clarification."""
        return bool(self.pending_options)

    def add(
        self,
        speaker: str,
        raw: str,
        semantic: str = "",
        language: str = "English",
        was_clarified: bool = False
    ):
        """Appends a new turn and updates concept activations with decay."""
        turn = Turn(
            speaker=speaker,
            raw=raw,
            semantic=semantic,
            language=language,
            was_clarified=was_clarified
        )
        self.turns.append(turn)

        # Apply exponential decay to prior context
        for k in list(self.context.keys()):
            self.context[k] = round(self.context[k] * self.decay_factor, 4)
            if self.context[k] < 0.05:
                del self.context[k]

        # Reinforce newly established semantic concept
        if semantic:
            clean_sem = semantic.strip().lower()
            self.context[clean_sem] = min(1.0, self.context.get(clean_sem, 0.0) + 0.75)
            self.resolved_concepts.append(clean_sem)
            self._update_domain(clean_sem)

    def _update_domain(self, concept: str):
        """Infers active shopkeeper domain from established concept."""
        concept = concept.lower()
        if concept in ["shirt", "tshirt", "shoes", "clothes", "hat"]:
            self.active_domain = "clothing"
        elif concept in ["blue", "black", "red", "white", "green", "color"]:
            self.active_domain = "color_selection"
        elif concept in ["small", "smalllittle", "biglarge", "large", "medium", "size"]:
            self.active_domain = "sizing"
        elif concept in ["bank", "storeorshop", "cellphone", "pen", "thankyou"]:
            self.active_domain = "transaction"

    def set_pending(self, question):
        """Sets the active pending question and extract selectable options."""
        if hasattr(question, "options"):
            self.pending_question = question
            self.pending_options = list(question.options)
            self.pending_candidate_keys = list(getattr(question, "candidate_keys", []))
        elif isinstance(question, (list, tuple)):
            self.pending_question = None
            self.pending_options = list(question)
            self.pending_candidate_keys = [str(o) for o in question]

    def resolve(self, answer: str) -> str:
        """
        Resolves customer clarification answer into the finalized semantic concept.
        Handles numeric option indices ('1', '2'), partial string matches, or yes/no.
        """
        raw_ans = str(answer).strip()
        ans_lower = raw_ans.lower()

        # Check for numeric choice ('1', '2', etc.)
        for i, opt in enumerate(self.pending_options, 1):
            if ans_lower == str(i):
                target = self.pending_candidate_keys[i - 1] if i - 1 < len(self.pending_candidate_keys) else opt
                self._clear_pending()
                return self._finalize_resolution(target)

        # Check for direct match in options or candidate keys
        for i, opt in enumerate(self.pending_options):
            opt_clean = opt.lower()
            if ans_lower == opt_clean or ans_lower in opt_clean or opt_clean in ans_lower:
                target = self.pending_candidate_keys[i] if i < len(self.pending_candidate_keys) else opt
                self._clear_pending()
                return self._finalize_resolution(target)

        # Handle Yes / No confirmation
        if ans_lower.startswith("yes") and self.pending_candidate_keys:
            target = self.pending_candidate_keys[0]
            self._clear_pending()
            return self._finalize_resolution(target)

        if ans_lower.startswith("no"):
            self._clear_pending()
            return "re-sign requested"

        # Fallback to literal answer
        self._clear_pending()
        return self._finalize_resolution(raw_ans)

    def _finalize_resolution(self, resolved_semantic: str) -> str:
        clean = resolved_semantic.strip().lower()
        # Clean prefix formatting if present (e.g. '1. Blue' -> 'blue')
        if len(clean) > 2 and clean[1] == "." and clean[0].isdigit():
            clean = clean[2:].strip()
        self.context[clean] = min(1.0, self.context.get(clean, 0.0) + 0.85)
        return clean

    def _clear_pending(self):
        self.pending_question = None
        self.pending_options = []
        self.pending_candidate_keys = []

    def reset(self):
        """Resets all conversational state."""
        self.turns.clear()
        self.context.clear()
        self._clear_pending()
        self.resolved_concepts.clear()
        self.active_domain = "general"
