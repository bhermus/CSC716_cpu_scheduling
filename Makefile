.PHONY: sim

.DEFAULT_GOAL := sim

sim:
	@echo "Enter simulation command:"
	@read -p ">> $$ sim " cmd && python main.py $$cmd
