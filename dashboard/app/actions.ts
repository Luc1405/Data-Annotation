"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { deleteRunById } from "../lib/db";

export async function deleteRunAction(formData: FormData) {
  const runId = String(formData.get("runId") ?? "").trim();

  if (runId) {
    await deleteRunById(runId);
    revalidatePath("/");
  }

  redirect("/");
}
