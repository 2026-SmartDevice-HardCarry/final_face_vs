async function sendInteraction() {
  try {
    await fetch("/api/interaction", { method: "POST" });
  } catch (e) {}
}