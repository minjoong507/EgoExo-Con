neg_occurrence = [
    "Is the event '{event}' absent from {st} to {ed} seconds in the video?",
    "Is the event '{event}' not present from {st} to {ed} seconds in the video?",
    "Does the event '{event}' not happen from {st} to {ed} seconds in the video?",
    "Is the event '{event}' missing from {st} to {ed} seconds in the video?"
]

pos_occurrence = [
    "Is the event '{event}' present from {st} to {ed} seconds in the video?",
    "Is the event '{event}' occurring from {st} to {ed} seconds in the video?",
    "Does the event '{event}' happen from {st} to {ed} seconds in the video?",
    "Is the event '{event}' included from {st} to {ed} seconds in the video?"
]

prompt = {
    "system": "You are a helpful assistant.",
    "grounding": "Give the query: '{event}', when does the described content occur in the video? "
                 "Please return its start and end time using 'start - end seconds'.",
    "pos": pos_occurrence,
    "neg": neg_occurrence,
    "add_detail": "Please answer with 'Yes' or 'No'.",
}