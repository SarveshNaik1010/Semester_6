const taskForm = document.getElementById("task-form");
const taskInput = document.getElementById("task-input");
const dueDateInput = document.getElementById("due-date");
const startInput = document.getElementById("start-datetime");
const endInput = document.getElementById("end-datetime");

const taskList = document.getElementById("task-list");
const dateList = document.getElementById("date-list");
const timeList = document.getElementById("time-list");
const taskCount = document.getElementById("task-count");

let totalTasks = 0;

taskForm.addEventListener("submit", function (e) {
  e.preventDefault();

  const taskText = taskInput.value.trim();
  const dueDate = dueDateInput.value;
  const startTime = startInput.value;
  const endTime = endInput.value;

  if (!taskText) return;

  // Remove empty messages
  removeEmptyMsg(taskList);
  removeEmptyMsg(dateList);
  removeEmptyMsg(timeList);

  // Task item
  const taskItem = document.createElement("li");
  taskItem.textContent = taskText;
  taskList.appendChild(taskItem);

  // Date info
  const dateItem = document.createElement("li");
  dateItem.textContent = `Task: "${taskText}" | Due: ${dueDate}`;
  dateList.appendChild(dateItem);

  // Time log
  const timeItem = document.createElement("li");
  timeItem.textContent = `Task: "${taskText}" | From ${startTime} → ${endTime}`;
  timeList.appendChild(timeItem);

  // Update counter
  totalTasks++;
  taskCount.textContent = totalTasks;

  // Reset form
  taskForm.reset();
});

function removeEmptyMsg(list) {
  const empty = list.querySelector(".empty-msg");
  if (empty) empty.remove();
}