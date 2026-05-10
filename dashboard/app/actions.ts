"use server";

import { existsSync } from "fs";
import path from "path";
import { spawn } from "child_process";
import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { deleteRunById } from "../lib/db";


function redirectWithRunError(message: string): never {
  redirect(`/run?error=${encodeURIComponent(message)}`);
}

function validateRunId(runId: string) {
  if (!/^[A-Za-z0-9._-]+$/.test(runId)) {
    redirectWithRunError("Run ID may only contain letters, numbers, dots, underscores, and dashes.");
  }
}

function validateScriptPath(scriptPath: string) {
  if (scriptPath === "layer_annotation.py") return scriptPath;
  if (/^annotation_versions\/[A-Za-z0-9._-]+\.py$/.test(scriptPath)) return scriptPath;
  redirectWithRunError("Selected script is not allowed.");
}

export async function startAnnotationRunAction(formData: FormData) {
  const runId = String(formData.get("runId") ?? "").trim();
  const scriptPath = validateScriptPath(String(formData.get("scriptPath") ?? "layer_annotation.py").trim());
  const model = String(formData.get("model") ?? "").trim();

  if (!runId) redirectWithRunError("Run ID is required.");
  validateRunId(runId);

  const repositoryRoot = path.resolve(process.cwd(), "..");
  const absoluteScriptPath = path.resolve(repositoryRoot, scriptPath);
  if (!absoluteScriptPath.startsWith(repositoryRoot) || !existsSync(absoluteScriptPath)) {
    redirectWithRunError("Selected script does not exist.");
  }

  const env: NodeJS.ProcessEnv = { ...process.env, RUN_ID: runId };
  if (model) env.OPENAI_MODEL = model;

  const child = spawn("python", [absoluteScriptPath, "--run-id", runId], {
    cwd: repositoryRoot,
    env,
    detached: true,
    stdio: "ignore",
  });
  child.unref();

  redirect(`/run?started=${encodeURIComponent(runId)}`);
}

export async function deleteRunAction(formData: FormData) {
  const runId = String(formData.get("runId") ?? "").trim();

  if (runId) {
    await deleteRunById(runId);
    revalidatePath("/");
  }

  redirect("/");
}
