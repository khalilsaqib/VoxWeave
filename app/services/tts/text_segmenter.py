import re


class TextSegmenter:
    def __init__(
        self,
        min_chars: int = 40,
        max_chars: int = 180,
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buffer = ""

    def add(self, text: str) -> list[str]:
        """
        Add new LLM text to the internal buffer.
        Returns any complete TTS-ready segments that can
        already be extracted.
        """
        if not text:
            return []

        self._buffer += text

        segments = []

        while True:
            segment = self._extract_segment()

            if segment is None:
                break

            segment = self.clean_text_for_tts(segment)

            if segment:
                segments.append(segment)

        return segments

    def flush(self) -> list[str]:
        """
        Return whatever text remains in the buffer.
        This is normally called when the LLM stream finishes.
        """
        text = self._buffer

        self._buffer = ""

        text = self.clean_text_for_tts(text)

        return [text] if text else []

    def _extract_segment(self) -> str | None:
        text = self._buffer

        if not text.strip():
            self._buffer = ""
            return None

        # --------------------------------------------------
        # 1. Detect numbered list items.
        #    Split the heading from the first item.
        # --------------------------------------------------

        list_pattern = r"(?m)^[ \t]*\d+[.)][ \t]+"
        list_matches = list(re.finditer(list_pattern, text))

        if list_matches:
            first = list_matches[0]

            # Return any introductory text before the first item.
            if text[:first.start()].strip():
                segment = text[:first.start()].strip()
                self._buffer = text[first.start():]
                return segment

            # If another item has started, the preceding item
            # is complete. Keep the next item for the next call.
            if len(list_matches) >= 2:
                second = list_matches[1]
                segment = text[:second.start()].strip()

                if segment:
                    self._buffer = text[second.start():]
                    return segment


        # --------------------------------------------------
        # 2. Find a real sentence boundary.
        #
        #    Requiring whitespace after punctuation avoids
        #    splitting inside decimals such as 543,480.66.
        # --------------------------------------------------
        sentence_pattern = r"[.!?](?=\s)"
        for match in re.finditer(sentence_pattern, text):
            end = match.end()

            # Don't split at the period in "1. CLIENT_ID".
            prefix = text[:end]

            if re.match(r"^\s*\d+[.)]\s*$", prefix):
                continue

            if end >= self.min_chars:
                segment = text[:end].strip()

                if segment:
                    self._buffer = text[end:].lstrip()
                    return segment
                

         # --------------------------------------------------
        # 3. Fallback for very long text.
        #    Never split on commas: they can be inside amounts.
        # --------------------------------------------------

        if len(text) >= self.max_chars:
            candidate = text[:self.max_chars]

            # Prefer a natural phrase boundary.
            boundaries = list(
                re.finditer(r"[;:]\s+", candidate)
            )

            if boundaries:
                boundary = boundaries[-1]
                end = boundary.end() - len(
                    boundary.group(0).lstrip()
                )
                if end >= self.min_chars:
                    segment = text[:end].strip()
                    self._buffer = text[end:].lstrip()
                    return segment

            # Otherwise split at whitespace, not inside a word.
            whitespace_matches = list(
                re.finditer(r"\s+", candidate)
            )

            if whitespace_matches:
                boundary = whitespace_matches[-1]
                end = boundary.start()

                if end >= self.min_chars:
                    segment = text[:end].strip()
                    self._buffer = text[boundary.end():]
                    return segment

        return None

    @staticmethod
    def clean_text_for_tts(text: str) -> str:
        if not text:
            return ""

        text = re.sub(
            r"\bCLIENT[\s_]*ID\b",
            "client ID",
            text,
            flags=re.IGNORECASE,
        )
        
        # Markdown links: retain only the readable label.
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Remove code markers.
        text = re.sub(r"`{1,3}", "", text)

        # Remove Markdown emphasis markers.
        text = re.sub(r"(\*\*|__|~~|\*|_)", "", text)

        # Remove heading markers.
        text = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]+", "", text)

        # Remove numbered-list prefixes.
        text = re.sub(r"(?m)^[ \t]*\d+[.)][ \t]+", "", text)

        # Normalize whitespace.
        text = re.sub(r"\s+", " ", text).strip()

        # Add natural sentence endings when an item ends in a number.
        if text and text[-1].isalnum():
            text += "."

        return text