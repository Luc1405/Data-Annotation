export function isPredictionMatch(goldValue: string | null, predictedValue: string | null) {
  const predicted = predictedValue?.trim() ?? "";
  if (!predicted || goldValue === null) return false;

  return goldValue
    .split(";")
    .map((label) => label.trim())
    .filter(Boolean)
    .includes(predicted);
}
