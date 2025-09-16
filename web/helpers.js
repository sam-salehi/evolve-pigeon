

export function areSameAnimals(a, b) {
    return (
        a.x === b.x &&
        a.y === b.y &&
        a.rotation === b.rotation &&
        a.satiation === b.satiation
    );
}
